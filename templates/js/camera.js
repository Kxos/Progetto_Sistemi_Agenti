// Grab elements, create settings, etc.
var video = document.getElementById('video');
var streamOject;

// Elements for taking the snapshot
var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
var video = document.getElementById('video');

// Elements for results
var image_result = document.getElementById('image_result');
var image_result_context = image_result.getContext('2d');
var heatmap = document.getElementById('heatmap');
var heatmap_context = heatmap.getContext('2d');

var STATE = "snap";

// Get access to the camera!
function openCam() {
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // Not adding `{ audio: true }` since we only want video now
        navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
            //video.src = window.URL.createObjectURL(stream);
            video.srcObject = stream;
            video.play();
            streamOject = stream;
        });
    }
}

function closeCam() {

    video.pause();
    video.srcObject = null;

    var track = streamOject.getTracks()[0];  // if only one media track

    track.stop();
}

openCam();

// Trigger photo take
document.getElementById("snap").addEventListener("click", function() {
	if (video.srcObject != null) {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
        context.drawImage(video,0,0);
        
        closeCam();

        $('#video').hide();
        $('#canvas').show();
    }
});

document.getElementById("retry").addEventListener("click", function() {
    if (STATE == "snap") {
        if (video.srcObject == null) {
            openCam();
            $('#canvas').hide();
            $('#video').show();
        }
    } else if (STATE == "analyze") {
        openCam();
        $('#image_result_cont').hide();
        $('#emotion_result_cont').hide();
        $('#heatmap_cont').hide();
        $('#canvas').hide();
        $('#snap').show();
        $('#analyze').show();
        $('#video').show();
    }
});

function toBlob(dataURL) {
    var array, binary, i;
    binary = atob(dataURL.split(',')[1]);
    array = [];
    i = 0;
    while (i < binary.length) {
      array.push(binary.charCodeAt(i));
      i++;
    }
    return new Blob([new Uint8Array(array)], {
      type: dataURL.split(',')[0].split(':')[1].split(';')[0]
    });
  }

  
document.getElementById("analyze").addEventListener("click", function() {
    // var server = "http://212.189.205.24/face-analyzer";
    var server = "https://www.intintlab.uniba.it/face-analyzer";
    
    if (video.srcObject == null) {
        var dataURL = canvas.toDataURL();

        var formData = new FormData();
        formData.append('image', toBlob(dataURL));
        formData.append('emotion', 'yes');
        formData.append('age', 'yes');
        formData.append('gender', 'yes');

        var response = $.ajax({
            url: server,
            data: formData,
            type: 'POST',
            processData: false,
            contentType: false
        })
        .done(function(data) {
            if ("error" in data) {
                $('#emotion_result').html(data["error"]);
                
                STATE = "analyze";
                $('#canvas').hide();
                $('#snap').hide();
                $('#analyze').hide();
                $('#image_result_cont').hide();
                $('#heatmap_cont').hide();
                $('#emotion_result_cont').show();
            } else {
                var image_result_img = new Image();
                image_result_img.onload = function() {
                    image_result.width = image_result_img.width;
                    image_result.height = image_result_img.height;
                    image_result_context.drawImage(image_result_img, 0, 0);
                };
                image_result_img.src = 'data:image/jpeg;base64,' + data["image_result"];
                
                
                
                var heatmap_img = new Image();
                heatmap_img.onload = function() {
                    heatmap.width = heatmap_img.width;
                    heatmap.height = heatmap_img.height;
                    heatmap_context.drawImage(heatmap_img, 0, 0);
                };
                heatmap_img.src = 'data:image/jpeg;base64,' +  data["heatmap"];

                $('#emotion_result').html("Emotion: <i>" + data["emotion"] + "</i>");
                
                STATE = "analyze";
                $('#canvas').hide();
                $('#snap').hide();
                $('#analyze').hide();
                $('#image_result_cont').show();
                $('#emotion_result_cont').show();
                $('#heatmap_cont').show();
            }
        })
        .fail(function() {
            console.log("no");
        });
    }

});



function getFrames(){
	if (video.srcObject != null) {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
        context.drawImage(video,0,0);
        
        $('#canvas').hide()
        // canvas.toDataURL('image/jpeg') E' il singolo Frame
	    // console.log(canvas.toDataURL('image/jpeg'));

        // Invia il Frame al Modello
        var socket = io('http://127.0.0.1:5000');
            socket.on('connect', function() {
                socket.emit('message', {data: canvas.toDataURL('image/jpeg')});
            });


       // Riceve i valori della Validazione
       socket.on("message", (data) => {
           console.log(data)
           // Stampa sull'html i valori della Validazione
           printResults(data);
      });
    
	}
	
}

var intervalId = window.setInterval(function(){
  /// call your function here
getFrames();
}, 2500);

function printResults(inputResults) {
    var results = inputResults;

    const obj = JSON.parse(results, function (key, value) {
        return value;
      });
            
    document.getElementById("results").innerHTML = "<strong>Emozione:</strong> " + obj.emotion_class + "<br><strong>Valenza:</strong> " + obj.Valenza + "<br><strong>Arousal:</strong> " + obj.Arousal;
}

    