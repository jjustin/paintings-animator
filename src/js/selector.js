var selected_img = ''
var selected_emoji = ''

function changeVideo() {
    console.log("Emoji:", selected_emoji, "Image:", selected_img)
    if (selected_emoji != '' && selected_img != '') {
        var output = document.getElementById("video")
        var source = document.getElementById("source")

        video.style.visibility = "visible"
        source.src = "output/" + selected_img + "_" + selected_emoji + ".mp4"
        output.load()
        output.play()
    }
}

function selectEmoji(emoji) {
    var exSelectedEmojis = document.getElementsByClassName("emojiImage selected")
    // Remove selected class
    for (let item of exSelectedEmojis) {
        item.classList.remove("selected")
    }
    emoji.classList.add("selected")
    selected_emoji = emoji.id

    var description = document.getElementById("description")
    if (selected_img == '') {
        description.innerHTML = "Sedaj izberi ali naloži sliko"
    }
    else {
        description.innerHTML = "OPIS SLIKE"
    }

    changeVideo()
}

