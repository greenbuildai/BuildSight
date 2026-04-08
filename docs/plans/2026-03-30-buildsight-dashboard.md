# BuildSight Dashboard Implementation Plan

> **For AI Agents:** REQUIRED SUB-SKILL: Use `executing-plans` (if available) to implement this task-by-task.

**Goal:** Build a frontend dashboard for BuildSight that presents live construction safety monitoring, AI alerts, and historical compliance metrics in a distinctive industrial utilitarian style.
**Architecture:** A standalone Vite React TypeScript app lives in `dashboard/` so it does not interfere with the Python pipelines in the repo root. The UI is composed from a small set of reusable components with design tokens defined in CSS variables and a dense monolithic layout optimized for operations monitoring.
**Tech Stack:** React, TypeScript, Vite, Vanilla CSS

### Task 1: Scaffold Isolated Frontend

**Files:**
- Create: `dashboard/*`

**Step 1: Create the app scaffold**
Run the Vite React TypeScript initializer in a new subfolder.

**Step 2: Install dependencies**
Install the generated package dependencies inside `dashboard/`.

**Step 3: Verify scaffold**
Run a production build before feature work starts.

### Task 2: Replace Starter With Dashboard Shell

**Files:**
- Modify: `dashboard/src/App.tsx`
- Modify: `dashboard/src/App.css`
- Modify: `dashboard/src/index.css`
- Modify: `dashboard/index.html`

**Step 1: Define design tokens**
Create the asphalt/concrete/orange color system and Outfit + JetBrains Mono typography.

**Step 2: Implement the monolithic layout**
Build the sidebar, top command bar, live camera stage, metric strip, trend section, and alert queue.

**Step 3: Add industrial motion and texture**
Use CSS-first scanlines, grid overlays, and restrained status motion.

### Task 3: Extract Reusable Monitoring Components

**Files:**
- Create: `dashboard/src/components/MetricCard.tsx`
- Create: `dashboard/src/components/LiveFeed.tsx`
- Create: `dashboard/src/components/AlertLog.tsx`

**Step 1: Metric cards**
Expose metric label, value, delta, status, and progress.

**Step 2: Live feed module**
Render the camera viewport, technical crosshairs, and AI detection overlays.

**Step 3: Alert log**
Render recent alerts with severity-coded treatment and timestamps.

### Task 4: Verify Output

**Files:**
- Test: `dashboard/`

**Step 1: Install frontend dependencies**
Use `npm install` inside `dashboard/`.

**Step 2: Run production build**
Use `npm run build` and fix any type or bundling issues.

**Step 3: Manual runtime check**
Use `npm run dev` to inspect the live UI locally.
