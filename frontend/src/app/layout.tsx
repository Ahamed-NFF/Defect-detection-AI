import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Defect Inspector",
  description:
    "GAN/Diffusion-augmented manufacturing defect detection — demo UI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
