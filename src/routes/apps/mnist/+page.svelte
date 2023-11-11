<script lang="ts">
  let gridSize = 28;
  let grid = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));
  let isMouseDown = false;

  function handleMouseDown() {
    isMouseDown = true;
  }

  function handleMouseUp() {
    isMouseDown = false;
  }

   function darkenCell(row, col) {
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
          if (newRow >= 0 && newRow < gridSize && newCol >= 0 && newCol < gridSize) {
            // Darken the neighboring cell to a lesser degree
            grid[newRow][newCol] = Math.min(grid[newRow][newCol] + 1, 10);
          }
        });
      });
    }
  }
</script>

<style>
    .mnist-grid{
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

<div class="mnist-grid w-[400px] md:w-[600px] mx-auto"
     on:mousedown={handleMouseDown}
     on:mouseup={handleMouseUp}>
    {#each grid as row, rowIndex}
        {#each row as cell, colIndex}
            <!-- svelte-ignore a11y-click-events-have-key-events -->
            <!-- svelte-ignore a11y-mouse-events-have-key-events -->
            <div class="cell h-[14.28px] md:h-[21.42px] "
                 style="background-color: hsl(0, 0%, {100 - cell * 15}%)"
                 on:mouseenter={() => darkenCell(rowIndex, colIndex)}
                 on:touchmove={() => darkenCell(rowIndex, colIndex)}
                 on:touchstart={() => darkenCell(rowIndex, colIndex)}

            >
            </div>
        {/each}
    {/each}
</div>
