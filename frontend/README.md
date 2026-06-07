# Frontend (demo UI)

Next.js app for the demo. Single page:
1. Drag-and-drop / upload a product image
2. Show classification result + confidence
3. Show Grad-CAM heatmap overlay (where the defect is)
4. Show LLaVA inspector description (if enabled)
5. Side gallery: synthetic defect images used in training (proves the GAN/diffusion worked)

Talks to the FastAPI backend at http://localhost:8000.

Owner: Member 4 (you). Scaffold with `npx create-next-app@latest .`
