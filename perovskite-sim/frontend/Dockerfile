FROM public.ecr.aws/docker/library/node:22-slim

WORKDIR /app

COPY perovskite-sim/frontend/package*.json ./
RUN npm ci

COPY perovskite-sim/frontend/ ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
