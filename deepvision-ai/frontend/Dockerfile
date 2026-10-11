# ==============================================================================
# DeepVision AI — Production Hardened Next.js Frontend Dockerfile
# Multi-stage build, minimal Alpine runtime, non-root user execution
# ==============================================================================

# Stage 1: Dependencies & Compilation
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies required for node-gyp if needed
RUN apk add --no-cache libc6-compat

COPY package*.json ./
RUN npm ci

# Copy configuration and source files
COPY tsconfig.json next.config.mjs tailwind.config.ts postcss.config.js ./
COPY src ./src

# Build production bundle
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Stage 2: Production Minimal Runtime
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000

# Create unprivileged system user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy built artifacts and runtime package definitions
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder --chown=nextjs:nodejs /app/next.config.mjs ./
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

CMD ["npx", "next", "start", "-p", "3000"]

