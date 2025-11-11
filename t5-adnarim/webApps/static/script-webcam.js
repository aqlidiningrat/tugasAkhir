var video = document.getElementById('videoFeed');
var canvas = document.createElement('canvas');
var context = canvas.getContext('2d');

const captureBtn = document.getElementById('captureBtn')

// Request access to the user's camera
navigator.mediaDevices.getUserMedia({video: true})
.then((stream) => {
  video.srcObject = stream;
  captureBtn.style.display = "inline-block";
})
.catch((error) => {
  console.error("Error accessing the camera: ", error);
});

function capture(dari, idWebcam){
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  // send the captured image data to the server
  canvas.toBlob((blob) => {
    const formData = new FormData();
    formData.append('imageBlob', blob, 'image_blob.jpg');
    fetch('/'+dari+'/'+idWebcam, {
      method: 'POST',
      body: formData
    })
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));
  }, 'image/jpg', 0.9);

  captureBtn.style.display = "none";
  // turn off webcam
  const stream = video.srcObject;
  const tracks = stream.getTracks();
  tracks.forEach(track => track.stop());
  video.srcObject = null;
  video.remove();

  // alert
  showAlert("Menyimpan gambar kedalam database..");
  function showAlert(message) {
  const alert= document.getElementById("alert");
  alert.style.display="block";
  alert.innerHTML = message;
  setTimeout(function(){
    alert.style.display="none";
    location.replace('/'+dari+'/'+idWebcam+'/potongGambar');
  },2500);
  };
};
