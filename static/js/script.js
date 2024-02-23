document.getElementById('startBtn').addEventListener('click', async function() {
    const button = document.getElementById('startBtn')
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
        } else if (count == -1) {
            countdown.textContent = ""
            overlay.removeAttribute("class", "overlay");
            clearInterval(interval);
            let time = 2;
            setInterval(function() {
                time--;
                if (time == 0) {
                    countdown.textContent = "Stop!";
                    overlay.setAttribute("class", "overlay");
                    var audio = new Audio('static/audio/stop.mp3');
                    audio.play().catch(error => console.error("Playback failed:", error));
                } else if (time == -1) {
                    setTimeout(function() {
                        window.location.href = 'result'
                    }, 100000);
                }
            }, 1000);
        }
    }, 1000);
});
