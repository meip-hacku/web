document.getElementById('splash-start').addEventListener('click', function() {
    this.style.display = 'none';
    
    // 音声ファイルのパス
    var audio = new Audio('static/audio/GanbariMascleTitle.wav');

    var img = document.getElementById('logoImage');
    img.style.display = 'block'; // 画像を表示
    img.style.opacity = 0; // 透明度を0に設定

    var opacity = 0;
    var audio = new Audio('/static/audio/GanbariMascleTitle.wav'); // 音声ファイルのパス
    var itr_num = 0;

    var intervalID = setInterval(function() {
        if (opacity < 1) {
            opacity += 0.006; // 透明度を徐々に上げる
            img.style.opacity = opacity;
            itr_num += 1
        } else {
            clearInterval(intervalID); // フェードイン完了後にインターバルをクリア
        }
        if (itr_num == 165) {
            audio.play().then(() => {
                console.log("Audio started successfully");
            }).catch(error => {
                console.error("Playback failed:", error);
            });
        }
        if (itr_num > 165) {
            clearInterval(intervalID); // インターバルをクリア
            setTimeout(function() {
                window.location.href = "/play"; // メインページへのパス
            }, 1700); //1500ミリ秒後に遷移
        }
    }, 10); // 10ミリ秒ごとに透明度を更新
});