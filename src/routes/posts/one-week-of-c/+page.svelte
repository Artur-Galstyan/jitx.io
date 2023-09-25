<script lang="ts">
    import HintBox from "$lib/components/HintBox.svelte";
</script>

<p>
    People say, that - at least once - every programmer should learn <i>some</i>
    <code>C</code>. If you familiarise yourself with the fundamentals, such as
    data structures, memory management and other low level stuff, it will allow
    you to grow your programming skills. And I agree. So, here I am, trying to
    learn <code>C</code>, the powerhouse of the
    <strike>cell</strike>, <i>uhm</i>, I mean of course modern technology.
</p>

<p>
    Before we dive into this, I'd like to mention that I have 0 experience
    writing code in <code>C</code>. All I know is that it's low level, there are
    pointers (I know they <i>point</i> to some memory address, but I don't know
    how to interact with them yet) and that if things are supposed to be fast in
    Python, <code>C</code> comes into play.
</p>

<HintBox
    content="I will try to make this as unfiltered as possible, while still providing an interesting reading experience. The days here are not consecutive; there can be some days in between. "
/>

<h1>Day 1 - Baby Steps</h1>
<p>
    My goal for now is to get <i>something</i> running, like for example
    printing <i>"hello world"</i> to the console. The first challenge is to
    download all I need to get started. After some googling, I found out that
    there is GCC and <code>clang</code>, which can compile my <code>C</code>
    code into something executable. I found this
    <a
        class="link"
        href="https://alibabatech.medium.com/gcc-vs-clang-llvm-an-in-depth-comparison-of-c-c-compilers-899ede2be378"
        >medium blog post</a
    >, which goes into quite some detail, but honestly, it's too detailed for me
    and I don't understand most of it, except for the conclusion: it probably
    won't matter to me and either will work. Since I heard that Jax uses
    <code>LLVM</code>
    under the hood and I heard of Chris Lattner before and how he and his team are
    currently building Mojo, I have at least a face behind <code>LLVM</code>, so
    I'll go with <code>clang</code>. By the way, my understanding so far is that
    <code>LLVM</code>
    has all the bits and pieces to make a compiler and that <code>clang</code>
    is simply one constilation of that which works with <code>C</code>. So, I
    went ahead and downloaded that using
    <code>brew</code>
</p>
<pre><code class="language-text">{`brew install clang`}</code></pre>
<p>
    Afterwards, I created a file ending in <code>.c</code> and got to work.
    Since I'm currently using VSCode (although I'm looking for a high-fps,
    vim-first, nice LSP support IDE and if I don't find one soon, I probably
    make one myself), I got prompted to install the <code>C</code>/<code
        >C++</code
    > extension. Here's my very first C program. Apparently, this code is already
    riddled with bugs:
</p>
<pre><code class="language-c"
        >{`
int main()
{
    printf("hello world?");
    return 0;
}
`}</code
    ></pre>
<pre><code class="language-text"
        >{`
clang day1.c 
day1.c:4:5: error: call to undeclared library function 'printf' with type 'int (const char *, ...)'; ISO C99 and later do not support implicit function declarations [-Wimplicit-function-declaration]
    printf("hello world?");
    ^
day1.c:4:5: note: include the header <stdio.h> or explicitly provide a declaration for 'printf'
1 error generated.
`}</code
    ></pre>
<p>
    So close! But the error message is quite useful and I need to include the <code
        ><stdio.h /></code
    > <i>header</i> to use <code>printf</code>.
</p>

<HintBox
    content={`By the way, the only reason I know that it's <code>printffffff</code> is because of how Jeff from <a class="link" href="https://fireship.io">Fireship</a> pronounced it in one of his 100s videos.`}
/>
<p>
    I've added that in and got no errors this time! But no console output
    either!
</p>
<pre><code class="language-c"
        >{`
#include <stdio.h>

int main()
{
    printf("hello world?");
    return 0;
}`}</code
    ></pre>
<p>
    Turns out, <code>clang</code> compiles it into an executable, but I still need
    to actually execute it to see something.
</p>
<pre><code class="language-text"
        >{`clang day1.c && ./a.out 
hello world?%`}</code
    ></pre>
<p>
    Great! It worked. I have no idea why there is a <i>"%"</i> sign there, but for
    now I'm happy to see something on the console! The next day, I'll try to implement
    matrix multiplication, but for now it's time to get some rest. All of this memory
    management takes its toll on you.
</p>
