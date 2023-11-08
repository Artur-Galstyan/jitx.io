import { lucia } from "lucia";
import { sveltekit } from "lucia/middleware";
import { prisma as client } from "$lib/server/prisma.server";
import { prisma } from "@lucia-auth/adapter-prisma";
import { dev } from "$app/environment";
import { apple, discord, github } from "@lucia-auth/oauth/providers";
import { GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET } from "$env/static/private";
import { env } from "$env/dynamic/private";

export const auth = lucia({
  env: dev ? "DEV" : "PROD",
  middleware: sveltekit(),
  adapter: prisma(client, {
    user: "user", // model User {}
    key: "key", // model Key {}
    session: "session", // model Session {}
  }),
  getUserAttributes: (data) => {
    return {
      comments: data.comments,
      reactions: data.reactions,
    };
  },
});

export const appleAuth = apple(auth, {
  clientId: env.APPLE_CLIENT_ID,
  redirectUri: env.APPLE_REDIRECT_URI,
  teamId: env.APPLE_TEAM_ID,
  keyId: env.APPLE_KEY_ID,
  certificate: env.APPLE_CERTIFICATE,
  scope: ["name", "email"],
  responseMode: "form_post",
});

export const githubAuth = github(auth, {
  clientId: GITHUB_CLIENT_ID,
  clientSecret: GITHUB_CLIENT_SECRET,
  scope: ["user"],
});

export const discordAuth = discord(auth, {
  clientId: env.DISCORD_CLIENT_ID,
  clientSecret: env.DISCORD_CLIENT_SECRET,
  redirectUri: env.DISCORD_REDIRECT_URI,
});

export type Auth = typeof auth;
