FROM nginx:alpine

# Copy web files to nginx html directory
COPY index.html /usr/share/nginx/html/
COPY app.js /usr/share/nginx/html/

# Copy nginx configuration if needed
# COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
