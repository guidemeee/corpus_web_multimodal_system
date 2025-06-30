import nest_asyncio
nest_asyncio.apply()  # 启用嵌套事件循环支持

import asyncio
import traceback
import gc
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import inspect
import aiohttp
from functools import wraps
import time
import random
from typing import Callable, List, Any, Dict, Tuple, Optional, Union
import logging
import sys


class AsyncTasks:
    def __init__(self):
        self._session = None
        self._semaphores = {}  # 缓存不同并发级别的信号量
    
    async def _ensure_session(self):
        """确保aiohttp会话已创建，使用连接池优化"""
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit=100,  # 连接池大小
                keepalive_timeout=60,  # 连接保持时间
                ssl=False  # 如果不需要SSL，可提高性能
            )
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session
    
    async def _close_session(self):
        """关闭aiohttp会话"""
        if self._session is not None:
            await self._session.close()
            self._session = None
    
    def _get_semaphore(self, max_workers: int) -> asyncio.Semaphore:
        """获取指定大小的信号量，如果已存在则重用"""
        if max_workers not in self._semaphores:
            self._semaphores[max_workers] = asyncio.Semaphore(max_workers)
        return self._semaphores[max_workers]
    
    def is_coroutine_function(self, func: Callable) -> bool:
        """检查函数是否为协程函数"""
        return inspect.iscoroutinefunction(func)
    
    def wrap(self, function: Callable) -> Callable:
        """智能包装函数，如果已经是协程函数则不变，否则包装成异步函数"""
        if self.is_coroutine_function(function):
            return function
        
        @wraps(function)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: function(*args, **kwargs))
        return wrapper
    
    async def process_single_task(self, semaphore: asyncio.Semaphore, idx: int, 
                                function: Callable, task: List, 
                                is_coro: bool, retry_count: int = 0) -> Tuple[int, Any]:
        """处理单个任务，带有重试逻辑和信号量控制"""
        async with semaphore:
            max_retries = 3
            backoff_factor = 1.5  # 指数退避因子
            
            for attempt in range(max_retries):
                try:
                    if is_coro:
                        result = await function(*task)
                    else:
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(None, lambda: function(*task))
                    return idx, result
                except Exception as e:
                    if attempt < max_retries - 1:
                        # 使用指数退避策略
                        wait_time = (backoff_factor ** attempt) * (0.1 + 0.1 * random.random())
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        if retry_count == 0:  # 只在第一次重试时打印详细错误
                            print(f"任务 {idx} 执行失败: {str(e)}")
                            traceback.print_exc()
                        return idx, None
    
    async def process_tasks(self, function: Callable, task_list: List, 
                           name: str = "", max_workers: int = 50, 
                           max_retries: int = 3, use_tqdm: bool = True) -> List[Any]:
        """并发处理多个任务，使用智能并发控制和批处理"""
        if not task_list:
            return []
            
        results = []
        result_dict = {}  # 用于跟踪结果，确保顺序正确
        
        try:
            # 检测函数是否为协程函数
            is_coro = self.is_coroutine_function(function)
            
            # 获取或创建信号量
            semaphore = self._get_semaphore(max_workers)
            
            # 动态调整工作线程数，避免创建过多任务
            optimal_workers = min(max_workers, len(task_list) * 2)
            
            # 准备所有任务
            all_tasks = []
            for i, task in enumerate(task_list):
                all_tasks.append(self.process_single_task(semaphore, i, function, task, is_coro))
            
            # 使用tqdm显示进度
            completed = 0
            failed_indices = set()
            
            if use_tqdm:
                progress_bar = tqdm(total=len(all_tasks), desc=name if name else function.__name__)
            
            # 使用as_completed获取结果，可以更快地处理已完成的任务
            for future in asyncio.as_completed(all_tasks):
                idx, result = await future
                completed += 1
                
                if use_tqdm:
                    progress_bar.update(1)
                
                if result is None:
                    failed_indices.add(idx)
                else:
                    result_dict[idx] = result
            
            if use_tqdm:
                progress_bar.close()
            
            # 重试失败的任务
            if failed_indices and max_retries > 0:
                print(f"初次执行有 {len(failed_indices)} 个任务失败，开始重试")
                
                # 重试逻辑
                retry_tasks = []
                for retry_attempt in range(max_retries):
                    if not failed_indices:
                        break
                        
                    retry_indices = list(failed_indices)
                    failed_indices = set()  # 重置，记录本轮仍然失败的任务
                    
                    # 准备重试任务
                    retry_tasks.clear()
                    for idx in retry_indices:
                        retry_tasks.append(
                            self.process_single_task(semaphore, idx, function, task_list[idx], is_coro, retry_attempt + 1)
                        )
                    
                    if use_tqdm:
                        retry_progress = tqdm(total=len(retry_tasks), desc=f"重试 {retry_attempt + 1}/{max_retries}")
                    
                    # 并发执行重试任务
                    for future in asyncio.as_completed(retry_tasks):
                        idx, result = await future
                        
                        if use_tqdm:
                            retry_progress.update(1)
                        
                        if result is None:
                            failed_indices.add(idx)
                        else:
                            result_dict[idx] = result
                    
                    if use_tqdm:
                        retry_progress.close()
                    
                    print(f"第 {retry_attempt + 1} 轮重试后，还有 {len(failed_indices)} 个任务失败")
                    
                    # 如果没有失败的任务，或者达到最大重试次数，退出循环
                    if not failed_indices or retry_attempt == max_retries - 1:
                        break
                    
                    # 增加重试间隔，避免频繁重试导致资源竞争
                    await asyncio.sleep(0.1 * (retry_attempt + 1))
            
            # 如果仍有失败的任务，记录警告
            if failed_indices:
                print(f"警告：经过 {max_retries} 次重试后，仍有 {len(failed_indices)} 个任务执行失败")
                # 对于失败的任务，填充None值
                for idx in failed_indices:
                    result_dict[idx] = None
            
            # 构建结果列表，保持原始顺序
            for i in range(len(task_list)):
                if i in result_dict:
                    results.append(result_dict[i])
                else:
                    # 这种情况理论上不应该发生，但为了健壮性添加此处理
                    results.append(None)
            
            return results
        
        except Exception as e:
            print(f"任务处理过程中发生异常: {str(e)}")
            traceback.print_exc()
            # 返回已完成的结果和剩余任务的None值
            for i in range(len(task_list)):
                if i not in result_dict:
                    result_dict[i] = None
            return [result_dict.get(i) for i in range(len(task_list))]
        
        finally:
            # 释放内存
            result_dict.clear()
            gc.collect()
    
    async def _submit_async(self, function: Callable, task_list: List, 
                           name: str = "", max_workers: int = 50, 
                           max_retries: int = 3, use_tqdm: bool = True) -> List[Any]:
        """内部异步提交方法，包含智能会话管理和错误处理"""
        try:
            # 确保拥有会话
            await self._ensure_session()
            
            # 执行任务
            start_time = time.time()
            results = await self.process_tasks(function, task_list, name, max_workers, max_retries, use_tqdm)
            execution_time = time.time() - start_time
            
            if name:
                print(f"任务组 '{name}' 执行完成，耗时: {execution_time:.2f}秒")
            
            return results
        except Exception as e:
            print(f"提交任务时发生异常: {str(e)}")
            traceback.print_exc()
            # 返回与task_list长度相同的None列表
            return [None] * len(task_list)
        finally:
            # 关闭会话
            await self._close_session()
    
    def submit(self, function: Callable, task_list: List, 
              name: str = "", max_workers: int = 50, 
              max_retries: int = 3, use_tqdm: bool = True) -> List[Any]:
        """同步入口点，运行异步任务处理
        
        Args:
            function: 要执行的函数
            task_list: 任务参数列表，每个元素是一个列表，会被解包传给function
            name: 任务组名称，用于进度显示
            max_workers: 最大并发任务数
            max_retries: 失败任务最大重试次数
            use_tqdm: 是否显示进度条
            
        Returns:
            按原始顺序排列的执行结果列表
        """
        # 智能调整worker数量，避免过多并发
        if max_workers > 100:
            print(f"警告: max_workers值过高({max_workers})，已调整为100以避免资源耗尽")
            max_workers = 100
        
        # 批量提交前的参数预检
        if not task_list:
            return []
            
        # 捕获任何可能的事件循环异常
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有事件循环中执行
                nest_asyncio.apply()
                return loop.run_until_complete(self._submit_async(
                    function, task_list, name, max_workers, max_retries, use_tqdm
                ))
            else:
                # 使用当前事件循环
                return loop.run_until_complete(self._submit_async(
                    function, task_list, name, max_workers, max_retries, use_tqdm
                ))
        except RuntimeError:
            # 如果无法获取事件循环，创建新的
            return asyncio.run(self._submit_async(
                function, task_list, name, max_workers, max_retries, use_tqdm
            ))
        except Exception as e:
            print(f"事件循环操作异常: {str(e)}")
            traceback.print_exc()
            return [None] * len(task_list)


def test(a=0, b=0):
    return a + b, a - b


if __name__ == "__main__":
    print(test(1, 2))
    submit_args = [[1, 2], [3, 4], [5, 6]] * 10
    
    # 使用新的AsyncTasks类
    tasks = AsyncTasks()
    results = tasks.submit(test, submit_args, name="测试任务", max_workers=5)
    print(results)
    
    # 显式释放资源
    tasks = None
    results = None
    gc.collect()