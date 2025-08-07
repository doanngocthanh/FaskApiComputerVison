class BusinessConfig:
    def __init__(self):
        self.company_name = "Devhub Việt Nam"
        self.company_logo = "https://example.com/logo.png"
        self.company_description = "Devhub Việt Nam is a leading technology company specializing in software development and IT solutions."
        self.company_website = "https://devhub.io.vn"
        self.company_address = "None"
        self.contact_email = "contact@devhub.io.vn"
        self.contact_phone = "0905 xxx 027"
        self.support_email = "support@devhub.io.vn"
        self.is_business_config = False
        self.sepay_config = {
            "sepay_id": "your_sepay_id",
            "sepay_secret": "your_sepay_secret"
        }
        self.payos_config = {
            "payos_id": "your_payos_id",
            "payos_secret": "your_payos_secret"
        }
        self.path_middleware = "src.middleware.auth_middleware.AuthMiddleware"
        self.path_swagger = "src.service.SwaggerConfigService.swagger_config_service"
        self.path_security = "src.middleware.security_middleware.SecurityMiddleware"
        self.path_logging = "src.middleware.logging_middleware.AdvancedLoggingMiddleware" 