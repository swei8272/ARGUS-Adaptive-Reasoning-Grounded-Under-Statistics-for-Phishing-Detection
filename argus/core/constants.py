"""
常量定义 - 特征桶、触发规则、特征索引映射
"""

# ==================== 特征桶定义 ====================
FEATURE_BUCKETS = {
    'URL_BASIC': list(range(0, 38)),      # URL基础特征
    'HTML_FORM': list(range(38, 57)),     # HTML表单特征
    'STATISTICAL': list(range(57, 67)),   # 统计聚合
    'INTERACTION': list(range(67, 82)),   # 特征交互
    'DOMAIN_RULES': list(range(82, 92)),  # 领域规则
    'TEMPORAL': list(range(92, 100))      # 时序特征
}

# ==================== 特征索引映射 ====================
FEATURE_INDEX_MAP = {
    # URL特征 (0-37)
    'url_length': 0,
    'domain_length': 1,
    'path_length': 2,
    'query_length': 3,
    'fragment_length': 4,
    'num_dots': 5,
    'num_hyphens': 6,
    'num_underscores': 7,
    'num_slashes': 8,
    'num_questionmarks': 9,
    'num_equals': 10,
    'num_ats': 11,
    'num_ampersands': 12,
    'num_digits': 13,
    'digit_ratio': 14,
    'has_ip_address': 15,
    'num_subdomains': 16,
    'domain_entropy': 17,
    'tld_length': 18,
    'is_suspicious_tld': 19,
    'domain_token_count': 20,
    'domain_has_digits': 21,
    'subdomain_has_www': 22,
    'domain_brand_similarity': 23,
    'punycode_domain': 24,
    'domain_registration_length': 25,
    'domain_age': 26,
    'path_depth': 27,
    'num_params': 28,
    'has_login_keyword': 29,
    'has_update_keyword': 30,
    'has_verify_keyword': 31,
    'has_account_keyword': 32,
    'has_secure_keyword': 33,
    'has_banking_keyword': 34,
    'url_shortening_service': 35,
    'has_suspicious_keywords': 36,
    'param_entropy': 37,

    # HTML特征 (38-56)
    'num_forms': 38,
    'num_input_fields': 39,
    'num_password_fields': 40,
    'num_hidden_fields': 41,
    'form_has_action': 42,
    'form_action_external': 43,
    'has_login_form': 44,
    'form_to_email': 45,
    'num_links': 46,
    'external_link_ratio': 47,
    'null_link_ratio': 48,
    'num_external_redirects': 49,
    'has_suspicious_links': 50,
    'favicon_external': 51,
    'num_images': 52,
    'external_image_ratio': 53,
    'num_iframes': 54,
    'num_scripts': 55,
    'external_script_ratio': 56,

    # 统计聚合 (57-66)
    'url_mean': 57,
    'url_std': 58,
    'url_max': 59,
    'url_min': 60,
    'url_nonzero_count': 61,
    'html_mean': 62,
    'html_std': 63,
    'html_max': 64,
    'html_min': 65,
    'html_nonzero_count': 66,

    # 特征交互 (67-81)
    'url_domain_length_ratio': 67,
    'path_url_length_ratio': 68,
    'domain_subdomain_product': 69,
    'dots_subdomains_product': 70,
    'hyphens_entropy_product': 71,
    'suspicious_tld_entropy_product': 72,
    'ip_forms_product': 73,
    'brand_similarity_suspicious_tld_product': 74,
    'password_form_ratio': 75,
    'hidden_form_ratio': 76,
    'external_link_forms_product': 77,
    'password_external_link_product': 78,
    'images_forms_ratio': 79,
    'images_external_link_product': 80,
    'login_high_external_link': 81,

    # 领域规则 (82-91)
    'rule_typosquatting': 82,
    'rule_homograph_attack': 83,
    'rule_phishing_kit': 84,
    'rule_credential_harvesting': 85,
    'rule_ip_login_form': 86,
    'rule_long_url_suspicious_tld': 87,
    'rule_high_entropy_multi_subdomain': 88,
    'rule_external_resources': 89,
    'rule_complex_subdomain_structure': 90,
    'rule_short_high_entropy_domain': 91,

    # 时序特征 (92-99)
    'no_domain_age': 92,
    'very_new_domain': 93,
    'new_domain': 94,
    'old_domain': 95,
    'no_registration_length': 96,
    'short_registration': 97,
    'long_registration': 98,
    'age_registration_ratio': 99
}

# ==================== 先验触发规则 ====================
PRIOR_TRIGGER_RULES = {
    'domain_brand_consistency': {
        'buckets': ['URL_BASIC', 'DOMAIN_RULES'],
        'conditions': [
            ('domain_entropy', '>', 3.5),
            ('num_subdomains', '>=', 2),
            ('is_suspicious_tld', '==', 1.0),
            ('rule_typosquatting', '==', 1.0)
        ],
        'aggregation': 'any'  # 满足任一条件即触发
    },

    'high_entropy_subdomain_check': {
        'buckets': ['URL_BASIC'],
        'conditions': [
            ('domain_entropy', '>', 3.5),
            ('num_subdomains', '>=', 2)
        ],
        'aggregation': 'all'  # 需要同时满足
    },

    'ip_punycode_risk_check': {
        'buckets': ['URL_BASIC'],
        'conditions': [
            ('has_ip_address', '==', 1.0),
            ('punycode_domain', '==', 1.0)
        ],
        'aggregation': 'any'
    },

    'domain_age_verification': {
        'buckets': ['TEMPORAL'],
        'conditions': [
            ('no_domain_age', '==', 1.0),
            ('very_new_domain', '==', 1.0),
            ('short_registration', '==', 1.0)
        ],
        'aggregation': 'any'
    },

    'form_action_consistency': {
        'buckets': ['HTML_FORM'],
        'conditions': [
            ('num_forms', '>', 0),
            ('form_action_external', '==', 1.0)
        ],
        'aggregation': 'all'
    },

    'credential_harvesting_check': {
        'buckets': ['HTML_FORM', 'DOMAIN_RULES'],
        'conditions': [
            ('num_password_fields', '>', 0),
            ('form_action_external', '==', 1.0),
            ('rule_credential_harvesting', '==', 1.0)
        ],
        'aggregation': 'any'
    },

    'external_resource_check': {
        'buckets': ['HTML_FORM'],
        'conditions': [
            ('external_script_ratio', '>', 0.5),
            ('external_link_ratio', '>', 0.4),
            ('favicon_external', '==', 1.0)
        ],
        'aggregation': 'any'
    },

    'redirect_chain_check': {
        'buckets': ['HTML_FORM'],
        'conditions': [
            ('num_external_redirects', '>', 0),
            ('null_link_ratio', '>', 0.3)
        ],
        'aggregation': 'any'
    },

    'interaction_evidence_reinforcement': {
        'buckets': ['INTERACTION'],
        'conditions': [
            ('login_high_external_link', '==', 1.0),
            ('password_external_link_product', '>', 0.3)
        ],
        'aggregation': 'any'
    },

    'login_external_link_priority': {
        'buckets': ['INTERACTION'],
        'conditions': [
            ('login_high_external_link', '==', 1.0)
        ],
        'aggregation': 'all'
    },

    'typosquatting_detection': {
        'buckets': ['DOMAIN_RULES'],
        'conditions': [
            ('rule_typosquatting', '==', 1.0)
        ],
        'aggregation': 'all'
    },

    'homograph_attack_detection': {
        'buckets': ['DOMAIN_RULES'],
        'conditions': [
            ('rule_homograph_attack', '==', 1.0)
        ],
        'aggregation': 'all'
    },

    'phishing_kit_pattern': {
        'buckets': ['DOMAIN_RULES'],
        'conditions': [
            ('rule_phishing_kit', '==', 1.0),
            ('rule_external_resources', '==', 1.0)
        ],
        'aggregation': 'any'
    },

    'credential_form_external_action': {
        'buckets': ['DOMAIN_RULES'],
        'conditions': [
            ('rule_credential_harvesting', '==', 1.0)
        ],
        'aggregation': 'all'
    },

    'complex_subdomain_structure': {
        'buckets': ['DOMAIN_RULES'],
        'conditions': [
            ('rule_complex_subdomain_structure', '==', 1.0)
        ],
        'aggregation': 'all'
    },

    'short_high_entropy_domain': {
        'buckets': ['DOMAIN_RULES'],
        'conditions': [
            ('rule_short_high_entropy_domain', '==', 1.0)
        ],
        'aggregation': 'all'
    },

    'temporal_consistency_check': {
        'buckets': ['TEMPORAL'],
        'conditions': [
            ('very_new_domain', '==', 1.0),
            ('short_registration', '==', 1.0)
        ],
        'aggregation': 'all'
    }
}