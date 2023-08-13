import { errorToast } from "$lib/utils/notifications";

export function toggleModal(dialogId: string) {
    let dialog: HTMLDialogElement = document.getElementById(
        dialogId
    ) as HTMLDialogElement;
    if (dialog === null) {
        errorToast("Error", "Dialog not found, id: " + dialogId);
        return;
    }
    if (dialog.open) {
        dialog.close();
    } else {
        dialog.showModal();
    }
}
