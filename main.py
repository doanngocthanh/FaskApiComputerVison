from config import RouterConfig,MiddlewareConfig
from fastapi import FastAPI
import uvicorn
app = FastAPI()
RouterConfig().include_routers(app, RouterConfig().api_dir, "src.router.api")
MiddlewareConfig.add_cors_middleware(app)
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)