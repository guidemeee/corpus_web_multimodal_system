import time
import random
import threading
from collections import deque, defaultdict

class ApiKeyManager:
    """API密钥管理器，提供智能负载均衡和错误处理，同时考虑RPM和TPM限制"""
    
    def __init__(self, api_keys, cooldown_time=1.0, rpm_limit=1000, tpm_limit=50000):
        """
        初始化API密钥管理器
        
        参数:
        - api_keys: API密钥列表，每个元素为[key, base_url, rpm]，其中key是API密钥，base_url是基础URL，rpm是该密钥的请求率限制
        - cooldown_time: 同一个密钥两次使用之间的最小冷却时间(秒)
        """
        self.api_keys = [key_info[0] for key_info in api_keys]  # 提取所有密钥
        self.cooldown_time = cooldown_time
        
        # 存储每个密钥对应的base_url
        self.key_urls = {key_info[0]: key_info[1] for key_info in api_keys}
        
        # 每个密钥的RPM限制
        self.key_rpm_limits = {key_info[0]: key_info[2] for key_info in api_keys}
        self.rpm_limit = rpm_limit  # 默认RPM限制，用于没有指定限制的密钥
        self.tpm_limit = tpm_limit  # 每分钟令牌限制(默认值)
        
        # 负载均衡相关阈值设置
        self.rpm_threshold = 0.80  # RPM使用率阈值，超过此值将转向其他密钥
        self.tpm_threshold = 0.80  # TPM使用率阈值，超过此值将转向其他密钥
        self.critical_threshold = 0.90  # 临界阈值，超过此值将阻塞密钥使用
        
        # 基本计数器
        self.usage_counts = {key: 0 for key in self.api_keys}  # 总使用次数
        self.error_counts = {key: 0 for key in self.api_keys}  # 错误次数
        self.last_used = {key: 0 for key in self.api_keys}     # 上次使用时间
        
        # 按分钟跟踪的请求计数(RPM)
        self.rpm_counts = {key: [] for key in self.api_keys}  # 每个密钥近一分钟内的请求时间戳列表
        
        # 按分钟跟踪的令牌数(TPM)
        self.tpm_records = {key: [] for key in self.api_keys}  # 每个密钥近一分钟内的(时间戳,令牌数)记录
        
        # 预估每个密钥的当前分钟使用情况
        self.estimated_tpm = {key: 0 for key in self.api_keys}  # 预估本分钟内还会使用的token数
        
        # 预估的每个请求平均token消耗（将根据历史数据动态调整）
        self.avg_tokens_per_request = {key: 1000 for key in self.api_keys}  # 默认初始值
        
        # 添加总token计数统计
        self.total_tokens = {key: {"input": 0, "completion": 0, "total": 0} for key in self.api_keys}
        
        self.lock = threading.Lock()
        self.key_status = {key: {
            "available": True,           # 密钥是否可用
            "last_error": None,          # 上次错误信息
            "blocked_until": 0,          # 阻塞到什么时间
            "rpm_reset_time": 0,         # RPM计数重置时间
            "tpm_reset_time": 0,         # TPM计数重置时间
            "minute_start": int(time.time() / 60) * 60,  # 当前分钟的开始时间戳
        } for key in self.api_keys}
        
        # 定期清理线程
        self.cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self.cleanup_thread.start()
        
        print(f"API密钥管理器已初始化，共 {len(self.api_keys)} 个密钥，RPM阈值: {self.rpm_threshold*100}%，TPM阈值: {self.tpm_threshold*100}%")
    
    def _periodic_cleanup(self):
        """定期清理过期数据，更新统计信息"""
        while True:
            try:
                with self.lock:
                    current_time = time.time()
                    current_minute = int(current_time / 60) * 60
                    
                    for key in self.api_keys:
                        # 检查分钟是否已经改变
                        if current_minute > self.key_status[key]["minute_start"]:
                            # 重置当前分钟
                            self.key_status[key]["minute_start"] = current_minute
                            # 重置预估TPM
                            self.estimated_tpm[key] = 0
                        
                        # 清理过期记录
                        self._cleanup_minute_counts(key, current_time)
                        
                        # 更新平均token消耗量（如果有足够的历史数据）
                        if len(self.tpm_records[key]) >= 5:
                            total_tokens = sum(tokens for _, tokens in self.tpm_records[key])
                            requests_count = len(self.rpm_counts[key])
                            if requests_count > 0:
                                self.avg_tokens_per_request[key] = max(500, total_tokens / requests_count)
                
                # 每10秒运行一次周期性清理
                time.sleep(10)
            except Exception as e:
                print(f"周期性清理线程出错: {e}")
                time.sleep(30)  # 出错后等待较长时间再重试
    
    def _cleanup_minute_counts(self, key, current_time):
        """清理超过一分钟的RPM和TPM计数"""
        one_minute_ago = current_time - 60
        # 移除超过一分钟的RPM时间戳
        self.rpm_counts[key] = [t for t in self.rpm_counts[key] if t > one_minute_ago]
        # 移除超过一分钟的TPM记录
        self.tpm_records[key] = [(t, tokens) for t, tokens in self.tpm_records[key] if t > one_minute_ago]
    
    def get_current_rpm(self, key, current_time):
        """获取当前每分钟请求数"""
        self._cleanup_minute_counts(key, current_time)
        return len(self.rpm_counts[key])
    
    def get_current_tpm(self, key, current_time):
        """获取当前每分钟令牌数（实际使用量+预估量）"""
        self._cleanup_minute_counts(key, current_time)
        actual_tpm = sum(tokens for _, tokens in self.tpm_records[key])
        # 返回实际使用量+预估量
        return actual_tpm + self.estimated_tpm[key]
    
    def get_key_rpm_limit(self, key):
        """获取指定密钥的RPM限制"""
        return self.key_rpm_limits.get(key, self.rpm_limit)
    
    def reserve_capacity(self, expected_tokens=None):
        """
        预留容量，返回有足够令牌容量的密钥
        
        参数:
        - expected_tokens: 预期使用的令牌数，如果为None则使用密钥的平均值
        
        返回:
        tuple: (key, base_url) - 选择的API密钥和对应的基础URL
        """
        with self.lock:
            current_time = time.time()
            
            # 构建每个密钥的状态信息
            key_status_info = {}
            for key in self.api_keys:
                # 获取该密钥的RPM限制
                key_rpm_limit = self.get_key_rpm_limit(key)
                
                # 如果没有提供预期令牌数，使用该密钥的平均值
                tokens_estimate = expected_tokens if expected_tokens is not None else self.avg_tokens_per_request[key]
                
                # 清理并计算当前RPM和TPM
                rpm = self.get_current_rpm(key, current_time)
                tpm = self.get_current_tpm(key, current_time)
                
                # 预估使用这个密钥后的RPM和TPM百分比
                rpm_after = (rpm + 1) / key_rpm_limit if key_rpm_limit > 0 else 0
                tpm_after = (tpm + tokens_estimate) / self.tpm_limit if self.tpm_limit > 0 else 0
                
                # 使用较高的百分比作为负载指标
                load_percentage = max(rpm_after, tpm_after)
                
                key_status_info[key] = {
                    "rpm": rpm,
                    "tpm": tpm,
                    "rpm_limit": key_rpm_limit,
                    "rpm_after": rpm_after,
                    "tpm_after": tpm_after,
                    "load_percentage": load_percentage,
                    "available": self.key_status[key]["available"],
                    "blocked_until": self.key_status[key]["blocked_until"],
                    "time_since_last_use": current_time - self.last_used[key],
                    "error_count": self.error_counts[key]
                }
            
            # 找出可用的密钥
            available_keys = []
            for key, status in key_status_info.items():
                # 密钥可用且未被阻塞，并且预估使用后仍有足够余量
                if (status["available"] and 
                    current_time > status["blocked_until"] and
                    status["rpm_after"] < self.critical_threshold and
                    status["tpm_after"] < self.critical_threshold):
                    available_keys.append(key)
            
            if not available_keys:
                # 如果没有立即可用的密钥，选择负载最低且已过冷却期的密钥
                candidates = [k for k in self.api_keys if self.key_status[k]["available"]]
                if candidates:
                    key = min(candidates, key=lambda k: key_status_info[k]["load_percentage"])
                else:
                    # 所有密钥都被阻塞，选择阻塞结束时间最早的
                    key = min(self.api_keys, key=lambda k: self.key_status[k]["blocked_until"])
                    # 如果阻塞时间未到，等待直到阻塞结束
                    wait_time = max(0, self.key_status[key]["blocked_until"] - current_time)
                    if wait_time > 0:
                        print(f"所有密钥暂时不可用，等待 {wait_time:.1f} 秒...")
                        time.sleep(wait_time)
            else:
                # 在可用密钥中，优先考虑负载均衡
                
                # 首先筛选出负载较低的密钥（低于阈值）
                low_load_keys = [k for k in available_keys if key_status_info[k]["load_percentage"] < self.rpm_threshold]
                
                if low_load_keys:
                    # 有低负载密钥可用
                    candidates = low_load_keys
                else:
                    # 所有密钥负载都较高，使用所有可用密钥
                    candidates = available_keys
                
                # 按负载百分比升序排序
                candidates = sorted(candidates, key=lambda k: key_status_info[k]["load_percentage"])
                
                # 增加随机选择因素，避免所有请求都选择同一个密钥
                # 从负载最低的前几个密钥中随机选择，实现更好的负载均衡
                top_count = min(3, len(candidates))
                top_candidates = candidates[:top_count]
                
                # 对负载最低的前3个密钥，按错误率和时间间隔加权选择
                weights = []
                for k in top_candidates:
                    # 计算权重：错误率越低，时间间隔越长，权重越高
                    error_weight = 1.0 / (self.error_counts[k] + 1)
                    time_weight = min(5.0, key_status_info[k]["time_since_last_use"]) / 5.0
                    # 最终权重是错误权重和时间权重的加权平均
                    weights.append((error_weight * 0.7) + (time_weight * 0.3))
                
                # 根据权重随机选择一个密钥
                key = random.choices(top_candidates, weights=weights, k=1)[0]
            
            # 更新使用信息
            self.usage_counts[key] += 1
            self.last_used[key] = current_time
            # 添加请求时间戳到RPM计数
            self.rpm_counts[key].append(current_time)
            
            # 预估本次请求的token使用量
            tokens_to_reserve = expected_tokens if expected_tokens is not None else self.avg_tokens_per_request[key]
            self.estimated_tpm[key] += tokens_to_reserve
            
            # 检查预留后是否接近限制
            self._check_limits_after_reservation(key, current_time)
            
            # 返回API密钥和对应的基础URL
            return key, self.key_urls.get(key, "")
    
    def _check_limits_after_reservation(self, key, current_time):
        """预留容量后检查是否接近限制，必要时阻塞密钥"""
        # 获取当前实际TPM和预估TPM
        actual_tpm = sum(tokens for _, tokens in self.tpm_records[key])
        estimated_total = actual_tpm + self.estimated_tpm[key]
        
        # 如果预估总量超过阈值，提前阻塞
        if estimated_total > self.tpm_limit * self.critical_threshold:
            # 计算到下一分钟开始的时间
            current_minute = int(current_time / 60) * 60
            seconds_to_next_minute = (current_minute + 60) - current_time
            
            # 设置阻塞时间
            self.key_status[key]["blocked_until"] = current_time + seconds_to_next_minute + 3  # 额外3秒缓冲
            # print(f"API密钥 ...{key[-8:]} TPM预估接近限制 ({estimated_total}/{self.tpm_limit})，暂时阻塞 {seconds_to_next_minute+3:.1f} 秒")
    
    def get_next_key(self):
        """
        获取下一个可用的API密钥，兼容旧接口
        
        返回:
        tuple: (key, base_url) - 选择的API密钥和对应的基础URL
        """
        return self.reserve_capacity()
    
    def record_token_usage(self, key, input_tokens, completion_tokens):
        """记录密钥的令牌使用情况"""
        with self.lock:
            current_time = time.time()
            # 记录总令牌使用量(输入+输出)
            total_tokens = input_tokens + completion_tokens
            
            # 从预估中减去，加入实际使用记录
            self.estimated_tpm[key] = max(0, self.estimated_tpm[key] - total_tokens)
            self.tpm_records[key].append((current_time, total_tokens))
            
            # 更新总token计数
            self.total_tokens[key]["input"] += input_tokens
            self.total_tokens[key]["completion"] += completion_tokens
            self.total_tokens[key]["total"] += total_tokens
            
            # 更新历史平均值
            if len(self.rpm_counts[key]) > 0:
                total_recorded = sum(tokens for _, tokens in self.tpm_records[key])
                avg = total_recorded / len(self.rpm_counts[key])
                # 平滑更新，避免大幅波动
                self.avg_tokens_per_request[key] = (self.avg_tokens_per_request[key] * 0.7) + (avg * 0.3)
            
            # 清理过期记录
            self._cleanup_minute_counts(key, current_time)
            
            # 检查是否接近TPM限制
            current_tpm = sum(tokens for _, tokens in self.tpm_records[key])
            if current_tpm > self.tpm_limit * self.critical_threshold:
                # 计算到下一分钟开始的时间
                current_minute = int(current_time / 60) * 60
                seconds_to_next_minute = (current_minute + 60) - current_time
                
                # 设置阻塞时间
                self.key_status[key]["blocked_until"] = current_time + seconds_to_next_minute + 3  # 额外3秒缓冲
                # print(f"API密钥 ...{key[-8:]} TPM接近限制 ({current_tpm}/{self.tpm_limit})，暂时阻塞 {seconds_to_next_minute+3:.1f} 秒")
                
                # 清空预估值
                self.estimated_tpm[key] = 0
    
    def report_error(self, key, error_type):
        """报告密钥使用时遇到的错误"""
        with self.lock:
            self.error_counts[key] += 1
            self.key_status[key]["last_error"] = str(error_type)
            
            error_str = str(error_type).lower()
            
            # 对于速率限制错误，暂时禁用该密钥较长时间
            if "rate limit" in error_str or "limit" in error_str or "429" in error_str or "上游负载已饱和" in error_str:
                # 确定是RPM还是TPM限制
                is_tpm_limit = "tpm" in error_str or "token" in error_str
                is_rpm_limit = "rpm" in error_str or "requests per minute" in error_str or "rate_limit" in error_str or "request limit" in error_str
                
                # 计算到下一分钟开始的时间
                current_time = time.time()
                current_minute = int(current_time / 60) * 60
                seconds_to_next_minute = (current_minute + 60) - current_time
                
                # 设置阻塞时间，TPM错误阻塞时间更长
                buffer_time = 10 if is_tpm_limit else 5
                cooldown_time = seconds_to_next_minute + buffer_time
                
                self.key_status[key]["blocked_until"] = current_time + cooldown_time
                
                # 清空RPM和TPM计数，下次重新计算
                self.rpm_counts[key] = []
                self.tpm_records[key] = []
                self.estimated_tpm[key] = 0
                
                # print(f"{error_str}\nAPI密钥 ...{key[-8:]} 暂时阻塞 {cooldown_time:.1f} 秒，直到下一分钟周期")
                
                # 如果是TPM限制，调整该密钥的平均token估计值
                if is_tpm_limit:
                    # 增加平均token估计，让系统更保守地使用此密钥
                    self.avg_tokens_per_request[key] *= 1.2
            
            # 对于其他类型的错误，采取相应策略
            elif "timeout" in error_str:
                # 超时错误，短暂冷却
                self.key_status[key]["blocked_until"] = time.time() + 5.0
            elif "server" in error_str or "500" in error_str:
                # 服务器错误，稍长时间冷却
                self.key_status[key]["blocked_until"] = time.time() + 10.0
            else:
                # 其他未知错误，短暂冷却
                self.key_status[key]["blocked_until"] = time.time() + 2.0
            
            # 重置过多错误的密钥的错误计数（避免永久惩罚）
            if self.error_counts[key] > 30:
                # 每30次错误后，将错误计数减半
                self.error_counts[key] = self.error_counts[key] // 2
    
    def set_limits(self, rpm_limit=None, tpm_limit=None, key_rpm_limits=None, rpm_threshold=None, tpm_threshold=None):
        """设置RPM和TPM限制及阈值"""
        with self.lock:
            if rpm_limit is not None:
                self.rpm_limit = rpm_limit
                print(f"已更新默认RPM限制为: {rpm_limit}")
            
            if tpm_limit is not None:
                self.tpm_limit = tpm_limit
                print(f"已更新TPM限制为: {tpm_limit}")
            
            if key_rpm_limits is not None:
                for key, limit in key_rpm_limits.items():
                    if key in self.api_keys:
                        self.key_rpm_limits[key] = limit
                print(f"已更新各密钥的RPM限制")
            
            if rpm_threshold is not None:
                self.rpm_threshold = max(0.5, min(0.95, rpm_threshold))
                print(f"已更新RPM阈值为: {self.rpm_threshold:.2f}")
            
            if tpm_threshold is not None:
                self.tpm_threshold = max(0.5, min(0.95, tpm_threshold))
                print(f"已更新TPM阈值为: {self.tpm_threshold:.2f}")
    
    def get_usage_distribution(self):
        """获取所有密钥的RPM和TPM分布情况"""
        with self.lock:
            current_time = time.time()
            distribution = {}
            
            for key in self.api_keys:
                key_rpm_limit = self.get_key_rpm_limit(key)
                rpm = self.get_current_rpm(key, current_time)
                actual_tpm = sum(tokens for _, tokens in self.tpm_records[key])
                estimated_tpm = self.estimated_tpm[key]
                total_tpm = actual_tpm + estimated_tpm
                
                distribution[key] = {
                    "rpm": rpm,
                    "actual_tpm": actual_tpm,
                    "estimated_tpm": estimated_tpm,
                    "total_tpm": total_tpm,
                    "base_url": self.key_urls.get(key, ""),
                    "rpm_limit": key_rpm_limit,
                    "avg_tokens_per_request": round(self.avg_tokens_per_request[key]),
                    "rpm_percentage": (rpm / key_rpm_limit) * 100 if key_rpm_limit > 0 else 0,
                    "tpm_percentage": (total_tpm / self.tpm_limit) * 100 if self.tpm_limit > 0 else 0,
                    "available": self.key_status[key]["available"],
                    "blocked_until": self.key_status[key]["blocked_until"],
                    "blocked_for": max(0, self.key_status[key]["blocked_until"] - current_time)
                }
            
            return distribution
    
    def get_health_report(self):
        """获取所有密钥的健康状态报告"""
        with self.lock:
            current_time = time.time()
            report = {
                "usage_counts": self.usage_counts.copy(),
                "error_counts": self.error_counts.copy(),
                "usage_distribution": self.get_usage_distribution(),
                "global_settings": {
                    "rpm_threshold": self.rpm_threshold,
                    "tpm_threshold": self.tpm_threshold,
                    "critical_threshold": self.critical_threshold,
                    "default_rpm_limit": self.rpm_limit,
                    "default_tpm_limit": self.tpm_limit
                },
                "key_status": {},
                "currently_available": [],
                "total_tokens": self.total_tokens.copy() # 添加总token统计到报告中
            }
            
            for key, status in self.key_status.items():
                key_rpm_limit = self.get_key_rpm_limit(key)
                rpm = self.get_current_rpm(key, current_time)
                
                actual_tpm = sum(tokens for _, tokens in self.tpm_records[key])
                estimated_tpm = self.estimated_tpm[key]
                total_tpm = actual_tpm + estimated_tpm
                
                report["key_status"][key] = {
                    "available": status["available"],
                    "base_url": self.key_urls.get(key, ""),
                    "last_error": status["last_error"],
                    "blocked_remaining": max(0, status["blocked_until"] - current_time),
                    "time_since_last_use": current_time - self.last_used[key],
                    "current_rpm": rpm,
                    "actual_tpm": actual_tpm,
                    "estimated_tpm": estimated_tpm,
                    "total_tpm": total_tpm,
                    "rpm_limit": key_rpm_limit,
                    "avg_tokens_per_request": round(self.avg_tokens_per_request[key]),
                    "rpm_percentage": (rpm / key_rpm_limit) * 100 if key_rpm_limit > 0 else 0,
                    "tpm_percentage": (total_tpm / self.tpm_limit) * 100 if self.tpm_limit > 0 else 0
                }
                
                # 添加当前可用的密钥列表
                if (status["available"] and 
                    current_time > status["blocked_until"] and
                    rpm < key_rpm_limit * self.critical_threshold and
                    total_tpm < self.tpm_limit * self.critical_threshold):
                    report["currently_available"].append(key)
            
            return report
            
    def get_token_usage(self, key=None):
        """
        获取指定密钥或所有密钥的token使用情况
        
        参数:
        - key: 可选，指定的API密钥。如果为None，则返回所有密钥的使用情况
        
        返回:
        dict: 包含输入、输出和总token使用量的字典
        """
        with self.lock:
            if key is not None:
                if key in self.total_tokens:
                    return self.total_tokens[key].copy()
                return None
            
            # 汇总所有密钥的使用情况
            all_keys_total = {"input": 0, "completion": 0, "total": 0}
            for key_data in self.total_tokens.values():
                all_keys_total["input"] += key_data["input"]
                all_keys_total["completion"] += key_data["completion"]
                all_keys_total["total"] += key_data["total"]
            
            return {
                "by_key": self.total_tokens.copy(),
                "total": all_keys_total
            }
            
    def get_total_tokens(self):
        """
        获取所有密钥token使用量的总和
        
        返回:
        int: 所有密钥的token总使用量
        """
        with self.lock:
            total = 0
            for key_data in self.total_tokens.values():
                total += key_data["total"]
            return total