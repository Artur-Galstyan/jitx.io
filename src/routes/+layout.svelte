<script lang="ts">
    import Navbar from "$lib/components/Navbar.svelte";
    import "../app.css";
    import {dev} from "$app/environment";
    import {inject} from "@vercel/analytics";
    import {currentUser} from "$lib/state/currentUser";
    import {page} from "$app/stores";
    import LoginDialog from "$lib/components/LoginDialog.svelte";
    import "@fontsource/fira-code";
    import "@fontsource/fira-code/700.css";
    import "@fontsource/ubuntu"; // Defaults to weight 400
    import "@fontsource/ubuntu/700.css"; // Defaults to weight 400
    import {invalidate, onNavigate} from "$app/navigation";
    import {transisting} from "$lib/state/transisting";
    import Footer from "$lib/components/Footer.svelte";
    import {onMount} from "svelte";


    export let data
    onNavigate((navigation) => {
        if (!document.startViewTransition) return;
        $transisting = true;
        return new Promise((resolve) => {
            document.startViewTransition(async () => {
                resolve();
                await navigation.complete;
                $transisting = false;
            });
        });
    });

    inject({mode: dev ? "development" : "production"});


    const {supabase, session} = data
    // $: ({ supabase, session } = data)

    onMount(() => {
        const {data} = supabase.auth.onAuthStateChange((event, _session) => {
            if (_session?.expires_at !== session?.expires_at) {
                invalidate('supabase:auth')
                $currentUser = _session?.user;
            }
        })

        return () => data.subscription.unsubscribe()
    })
    $currentUser = $page.data.session?.user;
</script>

<svelte:head>
    <title>JITx</title>
    <script defer src="https://www.google.com/recaptcha/api.js"></script>
</svelte:head>
<main class="mx-auto flex flex-col">
    <Navbar/>
    <div class="md:w-[90%] lg:w-[80%] xl:w-[70%] 2xl:w-[60%] md:mx-auto flex-1">
        <slot/>
    </div>
    <Footer/>
</main>

<LoginDialog/>
