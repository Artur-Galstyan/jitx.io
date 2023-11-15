import { writable } from "svelte/store";
import type { User } from "@prisma/client";

export const currentUser = writable<User | null>(null);
