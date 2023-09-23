import type { User } from "@prisma/client";
import { writable } from "svelte/store";

export const currentUser = writable<User | undefined>(undefined);
