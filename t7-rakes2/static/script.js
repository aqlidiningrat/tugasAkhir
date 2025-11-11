window.onload = function(){
  let canvas = document.getElementById("draw");
   let ctx = canvas.getContext("2d");

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.lineWidth = 200;
  ctx.globalCompositeOperation = "destination-out";

  // let isDrawing = false;
  // let lastX = 0;
  // let lastY = 0;

  function draw(e){
    // if (!isDrawing) return;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    // [lastX, lastY] = [e.offsetX, e.offsetY];
  }

  // canvas.addEventListener("mousedown", (e) => {
  //   isDrawing = true;
  //   [lastX, lastY] = [e.offsetX, e.offsetY];
  // });

  canvas.addEventListener("mousedown", function(envent){
    draw(event);
  });
  canvas.addEventListener("mousemove", function(envent){
    draw(event);
  });
  canvas.addEventListener("mouseout", function(event){
    draw(event);
  });

  // canvas.addEventListener("mouseup", draw);

  // canvas.addEventListener("mousemove", () => (isDrawing = false));
  // canvas.addEventListener("mouseup", () => (isDrawing = false));
  // canvas.addEventListener("mouseout", () => (isDrawing = false));
  // canvas.addEventListener("mousedown", () => (isDrawing = false));

}
