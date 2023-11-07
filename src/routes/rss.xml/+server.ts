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
    <rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"> 
    <channel>
    <title>JITx.io</title>
    <link>https://jitx.io</link>
    <description>AI and SE Projects by Artur A. Galstyan</description> 
    <atom:link href="/rss.xml" rel="self" type="application/rss+xml"/>
    ${posts
      .map((post) => {
        `
        <item>
          <title>${post.title}</title>
          <link>https://jitx.io/posts/${post.slug}</link>
          <description>${post.shortDescription}</description> 
          <pubDate>${post.updatedAt.toUTCString()}</pubDate>
          <author>Artur A. Galstyan</author>
          <guid isPermaLink="true">https://jitx.io/posts/${post.slug}</guid>
        </item>
       `;
      })
      .join("")}
   
  </channel>
</rss> 
    
    `.trim();

  return new Response(xml, { headers });
}) satisfies RequestHandler;
