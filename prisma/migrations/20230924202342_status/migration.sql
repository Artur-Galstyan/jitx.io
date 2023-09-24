/*
  Warnings:

  - You are about to drop the column `draft` on the `Post` table. All the data in the column will be lost.

*/
-- CreateEnum
CREATE TYPE "Status" AS ENUM ('DRAFT', 'PLANNED');

-- AlterTable
ALTER TABLE "Post" DROP COLUMN "draft",
ADD COLUMN     "status" "Status" NOT NULL DEFAULT 'DRAFT';
