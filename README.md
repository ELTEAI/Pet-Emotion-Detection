# 拉取镜像
docker pull wangxuhui0101/dog_emotion_ai_app_openai_deepseek_qwen:latest

# 运行镜像（按需添加挂载或端口）
docker run -it --rm wangxuhui0101/dog_emotion_ai_app_openai_deepseek_qwen:latest


# 常用命令：
构建镜像：
docker build -t dog_emotion_ai_app:latest .

删除容器：
docker rm -f dog_emotion_rag

重新运行容器：
docker run -dit --name dog_emotion_rag dog_emotion_ai_app:latest

进入容器：
docker exec -it dog_emotion_rag bash

 docker build -t dog_emotion_ai_app:latest .