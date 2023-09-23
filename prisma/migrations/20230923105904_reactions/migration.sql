/*
  Warnings:

  - You are about to drop the column `clapCount` on the `Comment` table. All the data in the column will be lost.
  - You are about to drop the column `fireCount` on the `Comment` table. All the data in the column will be lost.
  - You are about to drop the column `heartCount` on the `Comment` table. All the data in the column will be lost.
  - You are about to drop the column `partyCount` on the `Comment` table. All the data in the column will be lost.
  - You are about to drop the column `sadCount` on the `Comment` table. All the data in the column will be lost.
  - You are about to drop the column `thumbsDownCount` on the `Comment` table. All the data in the column will be lost.
  - You are about to drop the column `thumbsUpCount` on the `Comment` table. All the data in the column will be lost.
  - Made the column `draft` on table `Post` required. This step will fail if there are existing NULL values in that column.

*/
-- CreateEnum
CREATE TYPE "ReactionType" AS ENUM ('LIKE', 'DISLIKE', 'PARTY', 'CLAP', 'HEART', 'FIRE', 'SAD');

-- AlterTable
ALTER TABLE "Comment" DROP COLUMN "clapCount",
DROP COLUMN "fireCount",
DROP COLUMN "heartCount",
DROP COLUMN "partyCount",
DROP COLUMN "sadCount",
DROP COLUMN "thumbsDownCount",
DROP COLUMN "thumbsUpCount";

-- AlterTable
ALTER TABLE "Post" ALTER COLUMN "draft" SET NOT NULL;

-- CreateTable
CREATE TABLE "Reaction" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "type" "ReactionType" NOT NULL,
    "userId" TEXT NOT NULL,
    "commentId" TEXT NOT NULL,

    CONSTRAINT "Reaction_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "Reaction" ADD CONSTRAINT "Reaction_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Reaction" ADD CONSTRAINT "Reaction_commentId_fkey" FOREIGN KEY ("commentId") REFERENCES "Comment"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
