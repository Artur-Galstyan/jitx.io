-- AlterTable
ALTER TABLE "Comment" ALTER COLUMN "userId" DROP NOT NULL;

-- AlterTable
ALTER TABLE "Reaction" ALTER COLUMN "userId" DROP NOT NULL;
