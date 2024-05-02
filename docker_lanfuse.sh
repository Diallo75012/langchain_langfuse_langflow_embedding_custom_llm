#!/bin/bash

# Load environment variables
. ./.env

# Pull the latest image
docker pull langfuse/langfuse:2

# Run LangFuse with proper environment variables
docker run --name langfuse \
-e DATABASE_URL="postgresql://${DATABASE_USERNAME}:${DATABASE_PASSWORD}@${DATABASE_HOST}:${PSQL_PORT}/${DATABASE_NAME}" \
-e NEXTAUTH_URL="http://${DATABASE_HOST}:${PORT}" \
-e NEXTAUTH_SECRET="$NEXTAUTH_SECRET" \
-e SALT="$SALT" \
-p ${PORT}:${PORT} \
langfuse/langfuse
