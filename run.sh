docker build -t train_image .
docker run -d -v "$(pwd):/app" train_image
# docker run -d -v "$(pwd):/app" --rm train_image
# docker logs $(docker ps -aqf "ancestor=train_image")

