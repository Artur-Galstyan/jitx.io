import Swal from 'sweetalert2';

export const Toast = Swal.mixin({
	toast: true,
	position: 'top-end',
	showConfirmButton: false,
	timer: 5000,
	color: '#ffffff',
	background: '#000000'
});

export function showUploadingAlert() {
	Swal.fire({
		title: 'Uploading...',
		html: ` <progress class="progress progress-info w-56  my-auto mx-4" max="100" />`,
		icon: 'info',
		allowOutsideClick: false,
		allowEscapeKey: false,
		allowEnterKey: false,
		showConfirmButton: false,
		showCancelButton: false,
		showCloseButton: false,
		showClass: {
			popup: 'animate__animated animate__fadeIn animate__faster'
		},
		hideClass: {
			popup: 'animate__animated animate__fadeOut animate__faster'
		},
		customClass: {
			popup: 'bg-sweetalert'
		}
	});
}

export function showAlmostThereAlert() {
	Swal.fire({
		title: 'Almost there...',
		timerProgressBar: true,
		didOpen: () => {
			Swal.showLoading();
		},
		showClass: {
			popup: 'animate__animated animate__fadeIn animate__faster text-black'
		},
		hideClass: {
			popup: 'animate__animated animate__fadeOut animate__faster'
		},
		allowOutsideClick: false,
		allowEscapeKey: false,
		allowEnterKey: false,
		showConfirmButton: false,
		showCancelButton: false,
		showCloseButton: false,
		customClass: {
			popup: 'bg-sweetalert'
		}
	});
}

export function errorToast(title: string, text: string) {
	Toast.fire({
		icon: 'error',
		title: title,
		text: text,
		customClass: {
			popup: 'bg-sweetalert'
		}
	});
}

export function infoNotification(title: string, text: string) {
	Toast.fire({
		icon: 'info',
		title: title,
		text: text,
		customClass: {
			popup: 'bg-sweetalert'
		}
	});
}

export function successNotification(title: string, text: string) {
	Toast.fire({
		icon: 'success',
		title: title,
		text: text,
		customClass: {
			popup: 'bg-sweetalert'
		}
	});
}
