This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Collaboration Board

The Collaboration Board is a real-time collaborative whiteboard feature that requires a separate WebSocket server for real-time synchronization.

### Local Development

1. **Start the Next.js app:**
   ```bash
   npm run dev
   ```

2. **Start the WebSocket server (in a separate terminal):**
   ```bash
   npm run collab
   ```

3. Open http://localhost:3000/prototypes/collab-board

### Deploying on Vercel

Since Vercel doesn't support long-running WebSocket servers, you need to deploy the WebSocket server separately:

1. **Deploy WebSocket server to Railway, Render, or Heroku:**
   - Create a new Node.js project
   - Add the `collab-server.js` file
   - Deploy to Railway (https://railway.app), Render (https://render.com), or Heroku (https://heroku.com)

2. **Set the environment variable:**
   - In Vercel project settings, add `NEXT_PUBLIC_WS_URL` with your WebSocket server URL (e.g., `wss://your-app.railway.app`)

3. **Deploy the Next.js app to Vercel:**
   ```bash
   vercel deploy
   ```

### Collaboration Board Features
- Create/join rooms with unique room IDs
- Real-time drawing with pencil, shapes, text, and sticky notes
- Cursor presence showing other users' positions
- Export canvas as PNG
- Undo/Redo support

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
