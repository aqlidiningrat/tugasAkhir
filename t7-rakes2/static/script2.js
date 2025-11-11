var canvas = document.getElementById("signature-pad");
var ratio = Math.max(window.devicePixelRatio || 1, 1);
canvas.width = canvas.offsetWidth * ratio;
canvas.height = canvas.offsetHeight * ratio;
canvas.getContext("2d").scale(ratio, ratio);

var signaturePad = new SignaturePad(canvas, {backgroundColor: 'rgba(250, 250, 250)'});
document.getElementById('clear').addEventListener('click', function(){
  signaturePad.clear();
});

function captureSignature(namaTable, idDB){
  // convert the canvas image to a data URL
  const dataURL = canvas.toDataURL('image/jpeg');
  // send the captured image data to the server
  fetch('/tulisTandaTangan/'+namaTable+'/'+idDB, {
    method: 'POST',
    body: JSON.stringify({image_data: dataURL}),
    headers: {'Content-Type':'application/json'}
  }).then(response => {
    // Handle response as needed
    console.log(response);
    showAlert("klasifikasiKNN . . .", namaTable, idDB);

  }).catch(error => {
    console.error('Error captureSignature...', error);
    window.location.replace('/')
  });
};

function showAlert(message, namaTable, idDB) {
  canvas.remove();
  document.querySelector("h1").style.display = "none";
  document.getElementById("wrapper").style.display = "none";
  document.getElementById("clear").style.display = "none";
  document.getElementById('kenaliTandaTangan').style.display = "none";
  const alert= document.getElementById("alert");
  alert.style.display = "block";
  alert.style.marginTop = "10em"
  alert.innerHTML = message;
  setTimeout(function(){
    alert.style.display = "none";
    alert.style.display="block";
    alert.innerHTML = ". . .";
    window.location.replace('/klasifikasiKNN/'+namaTable+'/'+idDB);
  },1500);
};
