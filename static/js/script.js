document.getElementById('startBtn').addEventListener('click', function() {
    const button = document.getElementById('startBtn')
    button.disable = true;
    button.style.display = 'none'
    const countdown = document.getElementById('countdown');
    let count = 3;
    countdown.textContent = count;
    const interval = setInterval(function() {
        count--;
        if (count > 0) {
            countdown.textContent = count;
        } else if (count === 0) {
            countdown.textContent = "Start!"
        } else if (count == -1) {
            countdown.textContent = ""
            clearInterval(interval);
            let time = 2;
            setInterval(function() {
                time--;
                if (time == 0) {
                    window.location.href = 'result'
                }
            }, 1000);
        }
    }, 1000);
});
