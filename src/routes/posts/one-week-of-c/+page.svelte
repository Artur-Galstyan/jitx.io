<script lang="ts">
    import HintBox from "$lib/components/HintBox.svelte";
    import Katex from "$lib/components/Katex.svelte";
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

<h1>Day 2 - Matrices</h1>

<p>
    On my second day, I wanted to perform matrix multiplication, so get started,
    I tried to define a matrix. Unfortunately, I forgot to turn off GitHub
    Copilot and it spoiled it for me:
</p>
<pre><code class="language-c"
        >{`
#include <stdio.h>

int main(void)
{
    int matrix[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};
    printf("%d", matrix[1][1]);
    return 0;
}
`}</code
    ></pre>
<p>
    After turning off GH CP, and compiling and executing the code, it printed <code
        >5%</code
    >
    to the terminal. I still don't know where that percent sign is coming from, but
    for now, I'll just accept that it's just some kind of glitch in the matrix. After
    creating another matrix, it was time to write my first function other than the
    <code>main</code> function. Here's my first attempt, which was once again riddled
    with syntax errors:
</p>
<pre><code class="language-c"
        >{`int[][] matrixAddition(int a[][], int b[][])
{
    int matrix[3][3] = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
    };

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            matrix[i][j] = a[i][j] + b[i][j];
        }
    }

    return matrix;
}
`}</code
    ></pre>
<p>
    This was the point at which I decided that I need some help. Time to watch
    some tutorials. After a bit of time, I decided to take a step back. Instead
    of matrix addition, I'd start simpler by iterating over the array and
    outputting the values of the matrix to the console according to their
    position in the matrix. In Python, you would use <code>range(len(...))</code
    >
    to get the length of an array, but apparently not so in <code>C</code>. In
    <code>C</code> you have to calculate the total number of bytes used in an
    array and divide that by the number of bytes for the first element. In the
    matrix above, each <code>int</code> takes up 4 bytes, thus making the whole
    matrix 36 bytes because 4 * 9 = 36. To get the number of bytes of
    <i>something</i>
    in <code>C</code>, you use the <code>sizeof(...)</code> function, e.g.:
</p>
<pre><code class="language-c">{`printf("%lu", sizeof(b));`}</code></pre>
<p>So, to iterate over a 2d matrix, we'd do something like this:</p>
<pre><code class="language-c"
        >{`int a[3][3] = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}};
int rows = sizeof(a) / sizeof(a[0]);
int columns = sizeof(a[0]) / sizeof(a[0][0]);
for (int i = 0; i < rows; i++)
{
    for (int j = 0; j < columns; j++)
    {
        printf("[%d][%d]=%d\\n", i, j, a[i][j]);
    }
}
`}</code
    ></pre>
<pre><code class="language-text"
        >{`clang day2.c && ./a.out
[0][0]=1
[0][1]=2
[0][2]=3
[1][0]=4
[1][1]=5
[1][2]=6
[2][0]=7
[2][1]=8
[2][2]=9
`}</code
    ></pre>
<p>
    Armed with this new knowledge, I went back to my previous function, but this
    is where I noticed that <b>I can't even return a matrix</b>! Instead, I have
    to pre-allocate some memory and fill the values in the function itself.
    Then, I noticed another error. I wanted to make my function be as generic as
    possible, but that's not possible in <code>C</code>. You need to be specific
    in your arguments about how large the arrays are. This is quite the stark
    contrast to Python.
</p>

<pre><code class="language-c"
        >{`#include <stdio.h>

void matrixAddition(int rows, int columns, int a[rows][columns], int b[rows][columns], int result[rows][columns])
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
}

int main(void)
{
    int b[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};

    int a[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};
    int rows = sizeof(a) / sizeof(a[0]);
    int columns = sizeof(a[0]) / sizeof(a[0][0]);

    int result[rows][columns];
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            printf("[%d][%d]=%d\\n", i, j, result[i][j]);
        }
    }
    matrixAddition(rows, columns, a, b, result);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            printf("[%d][%d]=%d\\n", i, j, result[i][j]);
        }
    }
    return 0;
}
`}</code
    ></pre>
<pre><code class="language-text"
        >{`
clang day2.c && ./a.out
[0][0]=0
[0][1]=0
[0][2]=0
[1][0]=0
[1][1]=0
[1][2]=0
[2][0]=0
[2][1]=0
[2][2]=1871310864 // wtf is this?
[0][0]=2
[0][1]=4
[0][2]=6
[1][0]=8
[1][1]=10
[1][2]=12
[2][0]=14
[2][1]=16
[2][2]=18
`}</code
    ></pre>

<p>
    <code>C</code> likes to be as explicit as possible and seems to follow the
    functional programming paradigm. Now, it was time to implement matrix
    multiplication. Here's my final result. Note, that this is - by far - not
    the most efficient implementation of matrix multiplication, as this is <Katex
        math={"\\mathcal{O}(n^3)"}
    /> and most efficient algorithms can go up to <Katex
        math={"\\mathcal{O}(n^2.3728596)"}
    />.
</p>

<pre><code class="language-c"
        >{`
void matrixMultiplication(int aRows, int aColumns, int bColumns, int a[aRows][aColumns], int b[aColumns][bColumns], int result[aRows][bColumns])
{
    // a b c b -> a b - b c = a c
    // aRows aColumns X aColumns bRows

    for (int i = 0; i < aRows; i++)
    {
        for (int j = 0; j < bColumns; j++)
        {
            int temp = 0;
            for (int k = 0; k < aColumns; k++)
            {
                temp += a[i][k] * b[k][j];
            }
            result[i][j] = temp;
        }
    }
}


int main(void)
{

    int a[2][3] = {
        {1, 2, 3},
        {4, 5, 6}};

    int b[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}};

    int aRows = sizeof(a) / sizeof(a[0]);
    int aColumns = sizeof(a[0]) / sizeof(a[0][0]);

    int bRows = sizeof(b) / sizeof(b[0]);
    int bColumns = sizeof(b[0]) / sizeof(b[0][0]);

    int result[aRows][bColumns];

    matrixMultiplication(aRows, aColumns, bColumns, a, b, result);
    for (int i = 0; i < aRows; i++)
    {
        for (int j = 0; j < bColumns; j++)
        {
            printf("%d ", result[i][j]);
        }
        printf("\\n");
    }
    return 0;
}
`}</code
    ></pre>
<pre><code class="language-text"
        >{`
38 44 50 56 
83 98 113 128 
`}</code
    ></pre>
<p>
    Great! We have implemented matrix multiplication in <code>C</code>!
    Tomorrow, we will start using <code>C</code> in embedded systems!
</p>
