// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
import type {Comment, Reaction} from "@prisma/client";

declare global {
  namespace App {
    // interface Error {}
    // interface PageData {}
    // interface Platform {}

    interface Locals {
      auth: import("lucia").AuthRequest;
    }
  }
  declare namespace Lucia {
    type Auth = import("$lib/server/lucia.server.ts").Auth;
    type DatabaseUserAttributes = {
      comments: Comment[];
      reactions: Reaction[];
      username: string;
      image: string | null;
      email: string | null;
    };
    type DatabaseSessionAttributes = {};
  }
}

export {};
