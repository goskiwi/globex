FROM node:22-alpine AS build
WORKDIR /workspace
RUN npm install --global npm@11.6.2
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm test -- --run
RUN npm run build

FROM nginx:stable-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /workspace/dist /usr/share/nginx/html
EXPOSE 80
