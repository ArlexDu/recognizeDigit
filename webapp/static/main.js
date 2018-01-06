/**
 * Created by arlex on 2018/1/6.
 */
var canvas;
var context;
// 记录当前是否在画图
var isDrawing = false;

$(function () {
    //获取画布与绘图上下文
    canvas = $("#drawingCanvas").get(0);

    context = canvas.getContext('2d');
    context.strokeStyle = 'rgb(0,0,0)';
    context.lineWidth = 15;

    //画布添加鼠标事件
    canvas.onmousedown = startDrawing;
    canvas.onmouseup = stopDrawing;
    canvas.onmouseout = stopDrawing;
    canvas.onmousemove = draw;
})


//开始画图
function startDrawing(e) {
    isDrawing = true;
    //创建一个新的绘图路径
    context.beginPath();
    //把画笔移动到鼠标的位置
    context.moveTo(e.pageX-canvas.offsetLeft, e.pageY - canvas.offsetTop);
}

// 停止画图
function stopDrawing(e) {
    isDrawing = false;
}

//画图中
function draw(e) {
  if (isDrawing == true) {
    // 找到鼠标最新位置
    var x = e.pageX - canvas.offsetLeft;
    var y = e.pageY - canvas.offsetTop;
    // 画一条直线到鼠标最新位置
    context.lineTo(x, y);
    context.stroke();
  }
}

// 清除画布
function clearCanvas() {
  context.clearRect(0, 0, canvas.width, canvas.height);
}

// 保存画布
function saveCanvas() {
  var data = {'file':canvas.toDataURL()};
  $(".result").hide()
  $(".spinner").show();
  $.ajax({
      type:'POST',
      url:'/upload',
      data: data,
      success: function (response) {
            response = $.parseJSON(response)
            $(".spinner").hide();
            $(".result").show()
            $("#fnn").text("FNN预测结果:"+response.fnn)
            $("#cnn").text("CNN预测结果:"+response.cnn)
      }
  })
}
