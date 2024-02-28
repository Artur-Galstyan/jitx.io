<script lang="ts">
  import { onDestroy, onMount } from "svelte";

  let websocketUrl = "wss://api.jitx.io/ws/";
  let apiUrl = "/api/apps/mnist";

  let gridSize = 28;
  let grid = Array(gridSize)
    .fill(0)
    .map(() => Array(gridSize).fill(0));
  let isMouseDown = false;

  let prediction: number = -1;
  let websocket: WebSocket;
  let randomUserId = String(Math.floor(Math.random() * 1000000)) + "user";
  function clearGrid() {
    grid = Array(gridSize)
      .fill(0)
      .map(() => Array(gridSize).fill(0));
  }

  let predictInterval: any;
  onMount(() => {
    // predictInterval = setInterval(async () => {
    //     await predict();
    // }, 1000)
    console.log("userId", randomUserId);
    websocket = new WebSocket(websocketUrl + randomUserId);
    websocket.onopen = () => {
      console.log("opened");
    };
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log(data);
      if (data.type === "prediction") {
        prediction = data.prediction;
      }
    };
  });

  onDestroy(() => {
    clearInterval(predictInterval);
  });

  async function predict() {
    let gridArrayFlattened = grid.flat();
    let req = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        array: gridArrayFlattened,
        user_id: randomUserId,
      }),
    });

    let res = await req.json();
    let task_id = res;

    console.log("task_id", task_id);

    let checkInterval = setInterval(async () => {
      let taskReq = await fetch(apiUrl + "?task_id=" + task_id, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      let taskRes = await taskReq.json();
      console.log("taskRes", taskRes);
      if (taskRes.status === "SUCCESS") {
        prediction = taskRes.result;
        clearInterval(checkInterval);
      }
    }, 1000);
  }

  function handleMouseDown() {
    isMouseDown = true;
  }

  function handleMouseUp() {
    isMouseDown = false;
  }

  function darkenCell(row: number, col: number) {
    if (isMouseDown) {
      // Darken the clicked cell
      grid[row][col] = Math.min(grid[row][col] + 2, 10); // increment, but cap the value

      // Update the neighbors
      const neighborOffsets = [-1, 0, 1];
      neighborOffsets.forEach((dx) => {
        neighborOffsets.forEach((dy) => {
          if (dx === 0 && dy === 0) {
            // Skip the main cell
            return;
          }
          const newRow = row + dx;
          const newCol = col + dy;
          // Check if the new indices are within the bounds of the grid
          if (
            newRow >= 0 &&
            newRow < gridSize &&
            newCol >= 0 &&
            newCol < gridSize
          ) {
            // Darken the neighboring cell to a lesser degree
            grid[newRow][newCol] = Math.min(grid[newRow][newCol] + 1, 10);
          }
        });
      });
    }
  }
</script>

<div
  class="mnist-grid w-[400px] md:w-[600px] mx-auto"
  on:touchstart={handleMouseDown}
  on:touchend={handleMouseUp}
  on:mousedown={handleMouseDown}
  on:mouseup={handleMouseUp}
>
  {#each grid as row, rowIndex}
    {#each row as cell, colIndex}
      <div
        class="cell h-[14.28px] md:h-[21.42px]"
        style="background-color: hsl(0, 0%, {100 - cell * 25}%)"
        on:mouseenter={() => darkenCell(rowIndex, colIndex)}
        on:touchmove={() => darkenCell(rowIndex, colIndex)}
      ></div>
    {/each}
  {/each}
</div>
<div class="flex justify-center my-8 space-x-4">
  <button on:click={clearGrid} class="btn btn-secondary"> Clear </button>
  <button on:click={predict} class="btn btn-primary"> Predict </button>
</div>
{#if prediction !== -1}
  <div class="text-center">
    Prediction {prediction}
  </div>
{/if}

<style>
  .mnist-grid {
    display: grid;
    grid-template-columns: repeat(28, auto);
    grid-template-rows: repeat(28, auto);
    column-gap: 0 !important;
    row-gap: 0 !important;
  }
  .cell {
    border: 1px solid lightgray;
    margin: 0 !important;
    padding: 0 !important;
  }
</style>
