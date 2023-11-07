import type { RequestHandler } from "@sveltejs/kit";
import { prisma } from "$lib/server/prisma.server";

export const GET = (async ({ request, fetch }) => {
  const posts = await prisma.post.findMany({
    where: {
      NOT: {
        status: "HIDDEN",
      },
    },
  });

  // xml headers
  const headers = { "Content-Type": "application/rss+xml" };

  const xml = `
    <?xml version="1.0" encoding="UTF-8" ?>
    <rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom> 
    <channel>
  <title>JITx</title>
  <link>https://jitx.io</link>
  <description>AI and SE by Artur A. Galstyan</description>
  <channel> 
    <atom:link href="/rss.xml" rel="self" type="application/rss+xml"/>
    ${posts
      .map((post) => {
        `
        <item>
        <title>${post.title}</title>
        <link>https://jitx.io/posts/${post.slug}</link>
        <description>${post.shortDescription}</description> 
        <pubDate>${post.updatedAt}</pubDate>
        <author>Artur A. Galstyan</author>
        <guid>https://jitx.io/posts/${post.slug}</guid>
        </item>
       `;
      })
      .join("")}
   
  </channel>
</rss> 
    
    `.trim();

  return new Response(xml, { headers });
}) satisfies RequestHandler;
