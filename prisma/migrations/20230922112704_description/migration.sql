/*
  Warnings:

  - Made the column `thumbnail` on table `Post` required. This step will fail if there are existing NULL values in that column.

*/
-- AlterTable
ALTER TABLE "Post" ADD COLUMN     "thumbnailDescription" TEXT,
ALTER COLUMN "thumbnail" SET NOT NULL;
