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

_This example is a bit wishy-washy_

In recent times, more and more databases have emerged. So many, that is has become a meme whether there are more JavaScript frameworks or databases out there. I'm not so sure nowadays.

There can be a lot of hype around these technologies. But before following the trend blindly, ask yourself: Can I solve my task with just postgres? In 95% of times, the answer is **yes**. It may be boring, but that's ok! Postgres is one of the most battle tested databases out there and will do just fine. BuT cAn iT sCale? You have 1 active monthly user and that's yourself so you do you really need to care?

Keep it simple; use MySQL or postgres. But don't get me wrong, we engineers have a natural inclination to try out new technologies, which is a good thing! But if you're out there trying to deliver a product, then that's not the time to experiment.

## Example 3: Abstractions

So you're sitting in front of your laptop, dimmed lights, 4 mugs on the table which were once filled with coffee and you're writing some function and then it hits you. Your eyes widen, your heart starts racing. All of a sudden you can see years into the future and have noticed a case where you might need a more general function than this. You know the probability of this is slim to none, but you can't help yourself. It's now 8 mugs on the table, 3 a.m. in the morning and you've written the most abstract class imaginable, used all latin letters as type variables and were forced to add even some greek letters to make it work.

You lean back, satisfied with your work. A last sip of coffee. "Perfect".

It happens to most of us where we start prematurely abstracting our code and even though it feels like you're making progress, most of the time you're not and your code becomes harder to read and maintain. Instead of writing the _ThingBuilderFactory_ class, just write the _Thing_ class and be done with it. Premature abstractions in programming is like writing a detailled plan of how to learn the violin with all the music theory and a carefully crafted practice schedule, before even having touched the instrument. You feel like you're making progress, but the best way to learn the violin is to just start playing it. _Looks at violin in the corner of the room_.

## When to not strive for simplicity

Some problems are hard and require complex solutions. It's not always going to be possible to keep things simple. Sometimes, you'll need that new database, you'll need to add that other technology to your stack and sometimes you'll also need to write abstractions. The key is to only increase entropy when it's absolutely necessary and not just because you can.

There's a lot more on this topic to be said, but I'll leave it at that for now. I hope you enjoyed this post and that it made you think about your own projects and how you can keep the entropy low. Cheers.
