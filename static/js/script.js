document.addEventListener('DOMContentLoaded', function () {
    var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);
    var video = document.querySelector('video');

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            video.srcObject = stream;
            video.play();
        });

    video.addEventListener('play', function () {
        var canvas = document.createElement('canvas');
        (function loop() {
            if (!video.paused && !video.ended) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                var data = canvas.toDataURL('image/jpeg');
                socket.emit('frame', data);
                setTimeout(loop, 1000 / 30); // 30 fps
            }
        })();
    });

    const playing = () => {
        let audioPlaying = false;
        let audioRandThreshold = 0.3;
        let audioLen = 11;
        setInterval(function() {
            if (!audioPlaying && Math.random() < audioRandThreshold) {
                audioPlaying = true;
                var audioNum = Math.floor(Math.random() * audioLen);
                var audio = new Audio(`static/audio/good${String(audioNum).padStart(2, '0')}.wav`);
                audio.addEventListener('ended', function() {
                    audioPlaying = false;
                });
                audio.play();
            }
        }, 1000);
    }

    const button = document.getElementById('startBtn')
    button.addEventListener('click', async function() {
        const overlay = document.getElementById('overlay');
        button.disable = true;
        button.style.display = 'none'
        const countdown = document.getElementById('countdown');
        let count = 5;
        countdown.textContent = count;
        overlay.setAttribute("class", "overlay");
        const interval = setInterval(function() {
            count--;
            if (count > 0) {
                countdown.textContent = count;
            } else if (count === 0) {
                countdown.textContent = "Start!"
                var audio = new Audio('static/audio/start.mp3');
                audio.play().catch(error => console.error("Playback failed:", error));
                // 開始をサーバーに通知
                fetch('/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        start: true
                    })
                })
                // サーバーから終了の合図が来るのを検知
                socket.on('finish', function() {
                    countdown.textContent = "Finish!\nお疲れ様でした！";
                    overlay.setAttribute("class", "overlay");
                    var audio = new Audio('static/audio/stop.mp3');
                    audio.play().catch(error => console.error("Playback failed:", error));
                    setTimeout(() => {
                        window.location.href = '/result';
                    }, 2000);
                })
            } else if (count == -1) {
                countdown.textContent = ""
                overlay.removeAttribute("class", "overlay");
                clearInterval(interval);
                playing();
            }
        }, 1000);
    });
});
