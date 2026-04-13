# Backend API & Celery

## FastAPI
- main.py: 提供 /api/jv, /api/impedance, /api/degradation 同步接口
- requirements.txt: FastAPI 依赖

## Celery
- celery_app.py: 提供异步仿真任务（J-V、阻抗、degradation）
- requirements-celery.txt: Celery/Redis 依赖

## 运行说明
- 启动API: `uvicorn main:app --reload`
- 启动Celery: `celery -A celery_app.celery_app worker --loglevel=info`
- 需本地Redis服务

## 后续扩展
- 可将API切换为异步提交任务，前端可查询任务状态/结果。
