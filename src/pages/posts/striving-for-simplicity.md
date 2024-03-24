---
    title: Striving for Simplicity and so Should You
    pubDate: 2024-03-24
    description: "A blog post about striving for simplicity in software development and how it can help you in the long run."
    tags: ["simplicity"] 
    layout: ../../layouts/blogpost.astro
---

# Striving for Simplicity and so Should You

2024-03-24

## Contents

## Introduction

This blogpost will vary a lot from my previous ones, as this will be much more subjective and opionated -- so if that isn't for you, then you have come to the wrong place. _Ye be warned!_

Here are two of my (now new) oppinions.

1. **Oppose new technologies for as long as possible**
2. **Oppose abstractions in your code as long as you can**

In essence, both of these boil down to the main take:

**Strive for simplicity**

I have found myself time and time again doing the exact opposites of both of those but from this day onward, no more! The remainder of this post will consist of anecdotal evidence to further my case and blissful ignorance of any opposing arguments.

## Example 1: This Blog

This blog used to look different. It looked like this before:

![Before](/posts/striving-for-simplicity/before.webp)

It's still [online here](https://jitx-pk3im24tq-artur-galstyan.vercel.app/).

Pretty cool right? Look at all those colors, emojis, buttons and even the thumbnails. I even implemented a search field that does fuzzy search using [fuse.js](https://www.fusejs.io/).

I even implemented my own comment section with multiple login providers, just because I didn't want to use [utterences](https://github.com/utterance/utterances) _just because it says "powered by utteranc.es"_. Imagine that.

![Comments](/posts/striving-for-simplicity/comments.webp)

The above image is emblematic of the whole problem.

**No one cared**

I solved a problem, which didn't even exist and introduced Supabase to handle all the auth stuff. As a result, I had to maintain one more tool for exactly 0 users.

I used SvelteKit to write my blogposts in HTML, instead of Markdown (because I couldn't get [MDsveX](https://github.com/pngwn/MDsveX) to run due to skill issues on my end), so it quickly became needlessly tedious to write posts. And as an engineer, if something is tedious, then hell has to enter an ice age before I lay a finger on that.

So, what's the solution here? Let's apply the main take: _strive for simplicity_. Simplicity in this case, means to remove everything non-essential to blogging.

Do you need a comment section for 20 weekly visitors (where you make up at least half of those visits)? No.

Do you _really_ need thumbnails, slowing down your site's loading time while not providing any real value? Probably not.

Do you really need a search bar to search across 3 pages of posts? No.

Do you even need pagination? No. Just list them out.

Do you need user management (and Supabase) for your read-only blog? I can't believe past-me answered this question with a 'Yes'. Of course not!

So instead of SvelteKit, I went with Astro, because it would allow me to write Markdown (with LaTeX math) out of the box (or with very minor configuration). I then added TailwindCSS and DaisyUI and decided to use the simplest bit of styling I possibly can and **not** to dive into the _styling rabbithole_.

The end result is a much simpler codebase, where I just have to add a new Markdown file and just start writing.

![Files](/posts/striving-for-simplicity/files.webp)

Lovely. Arguably, the blog is _uglier_ now, but that's ok. Its goal is to transfer knowledge from my brain through some text into your brain and not allow you to login using Apple. <sub><sup>I still can't believe I thought that was a good idea.</sup></sub>

## Example 2: Databases

In recent times, more and more databases have emerged. So many, that is has become a meme whether there are more JavaScript frameworks or databases out there. I'm not so sure nowadays.

There can be a lot of hype around these technologies. But before following the trend blindly, ask yourself: Can I solve my task with just postgres? In 95% of times, the answer is **yes**. It may be boring, but that's ok! Postgres is one of the most battle tested databases out there and will do just fine. BuT cAn iT sCale? First, you have 1 active monthly user and that's yourself so you do you really need to care?

Keep it simple. Use MySQL or postgres.

TBC...
