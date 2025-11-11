function showAlert(message, namaTable, idDB) {
  document.querySelector("h1").style.display = "none";
  document.getElementById("drop-area").style.display = "none";
  document.getElementById("img-view2").style.display = "none";
  document.getElementById('kenaliTandaTangan').style.display = "none";
  const alert= document.getElementById("alert");
  alert.style.display = "block";
  alert.style.marginTop = "10em";
  alert.innerHTML = message;
  setTimeout(function(){
    alert.style.display = "none";
    alert.style.display="block";
    alert.innerHTML = ". . .";
    location.replace('/klasifikasiKNN/'+namaTable+'/'+idDB);
  },1500);
};
