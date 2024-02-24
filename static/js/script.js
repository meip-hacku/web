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
        const interval = setInterval(function() {
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
        return interval;
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
        const countarea = document.getElementById('count-area');
        const remainCount = document.getElementById('remain-count');
        let interval2 = null;
        const interval = setInterval(function() {
            count--;
            if (count > 0) {
                countdown.textContent = count;
            } else if (count === 0) {
                countdown.textContent = "Start!"
                var audio = new Audio('static/audio/start.mp3');
                audio.play().catch(error => console.error("Playback failed:", error));
                countarea.style.display = 'flex';
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
                socket.on('squat', function(count) {
                    if(count < 0) return;
                    remainCount.textContent = 5 - count;
                });
                // サーバーから終了の合図が来るのを検知
                socket.on('finish', function() {
                    countdown.innerHTML = "Finish!<br />お疲れ様でした！";
                    countdown.style.fontSize = "5em";
                    overlay.setAttribute("class", "overlay");
                    var audio = new Audio('static/audio/stop.mp3');
                    audio.play().catch(error => console.error("Playback failed:", error));
                    clearInterval(interval2);
                    setTimeout(() => {
                        window.location.href = '/result';
                    }, 5000);
                })
            } else if (count == -1) {
                countdown.textContent = ""
                overlay.removeAttribute("class", "overlay");
                clearInterval(interval);
                interval2 = playing();
            }
        }, 1000);
    });
});
