async function getArray() {
    return await fetch("images").then(response => response.json())
}

async function loadImages() {
    var obj = await getArray()
    var arr = obj.images
    var processing = obj.processing
    console.log(obj)
    for (img of processing) {
        const image = document.createElement('img')
        image.src = '/images/processing/' + img
        image.classList.add("processing")
        full_name = img
        img_n = img.split('.')
        image.id = img_n[0]
        image.name = img_n[0]

        document.querySelector('#image-picker').appendChild(image)
        // keep checking if file exists
        let checkForChange = async () => {
            let exists = await fetch(`exists/${full_name}`).then(x => x.text()).then(x => x == 'true')
            console.log(exists)
            if (exists) {
                image.classList.remove("processing")
                image.classList.add("image")
                image.addEventListener('click', () => {
                    onImageClick(image)
                })
            }
            else {
                setTimeout(checkForChange, 5000)
            }
        }
        setTimeout(checkForChange, 5000)
    }

    for (img of arr) {
        const image = document.createElement('img')
        image.src = 'images/' + img
        image.classList.add("image")
        img_n = img.split('.')
        image.id = img_n[0]
        image.name = img_n[0]

        document.querySelector('#image-picker').appendChild(image)
    }

    var description = document.getElementById("description")
    description.innerHTML = "Izberi sliko ali emotikon"

}

loadOnImageClick = () => {
    images = document.getElementsByClassName('image')
    for (let image of images) {
        if (image.id == "addImage") {
            image.addEventListener('click', addNewImage)
            continue;
        }
        image.addEventListener('click', () => {
            onImageClick(image)
        })
    }
}

onImageClick = (image) => {
    var exSelectedImages = document.getElementsByClassName("image selected")
    // Remove selected class
    for (let item of exSelectedImages) {
        item.classList.remove("selected")
    }
    image.classList.add("selected")
    selected_img = image.id

    var description = document.getElementById("description")
    if (selected_emoji == '') {
        description.innerHTML = "Sedaj izberi emotikon"
    }
    else {
        description.innerHTML = "OPIS SLIKE"
    }

    let overlay = document.createElement('div')
    overlay.classList.add("overlay")
    overlay.innerHTML = "Selected"
    image.appendChild(overlay)

    changeVideo()
}

addNewImage = () => {
    console.log("Opening file")
    var input = document.createElement('input');
    input.type = 'file';
    // Handle file upload

    input.onchange = async e => {
        var file = e.target.files[0];
        const formData = new FormData()
        formData.append('file', file)

        response = await fetch("images", { method: "POST", body: formData }).then(x => x.json())

        if (response.error) {
            alert("Error: " + response.error)
            return;
        }

        location.reload()
    }

    input.click();
}

// Load images and add listeners to each of them
loadImages().then(loadOnImageClick)
