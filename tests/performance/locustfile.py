from locust import HttpUser, task, between

class OpinionMarketUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def health_check(self):
        """Health check endpoint"""
        self.client.get("/health")
    
    @task(2)
    def metrics(self):
        """Metrics endpoint"""
        self.client.get("/metrics")
    
    @task(1)
    def root(self):
        """Root endpoint"""
        self.client.get("/")
    
    @task(1)
    def docs(self):
        """Documentation endpoint"""
        self.client.get("/docs")
    
    @task(1)
    def api_root(self):
        """API root endpoint"""
        self.client.get("/api/v1/")
    
    @task(1)
    def api_health(self):
        """API health endpoint"""
        self.client.get("/api/v1/health")
