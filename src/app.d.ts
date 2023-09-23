// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
declare global {
    namespace App {
        // interface Error {}
        // interface Locals {}
        // interface PageData {}
        // interface Platform {}

        interface Session {
            user?: {
                id: string;
                name: string;
                email: string;
                image: string;
                emailVerified: Date;
            };
            expires: string; // ISODateString
        }
    }
}

export {};
