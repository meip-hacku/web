document.getElementById('splash-start').addEventListener('click', function() {
    this.style.display = 'none';
    
    // 音声・画像ファイルのパス
    var audio = new Audio('static/audio/GanbariMascleTitle.wav');
    var img = document.getElementById('logoImage');
    
    img.style.display = 'block'; // 画像を表示
    img.style.opacity = 0; // 透明度を0に設定

    var opacity = 0;
    var itr_num = 0;

    var intervalID = setInterval(function() {
        if (opacity < 1) {
            opacity += 0.006; // 透明度を徐々に上げる
            img.style.opacity = opacity;
            itr_num += 1
        } else {
            clearInterval(intervalID); // フェードイン完了後にインターバルをクリア
        }
        if (itr_num == 165) { // 程よいところで音声を再生
            audio.play().then(() => {
                console.log("Audio started successfully");
            }).catch(error => {
                console.error("Playback failed:", error);
            });
        }
        if (itr_num > 165) { // 自動で計測画面へ遷移
            clearInterval(intervalID); // インターバルをクリア
            setTimeout(function() {
                window.location.href = "/play"; // メインページへのパス
            }, 1700);
        }
    }, 10); // 10ミリ秒ごとに透明度を更新
});