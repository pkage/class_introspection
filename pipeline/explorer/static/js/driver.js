app = new Vue({
    el: '#app',
    data: {
        experiment_list: [],
        epsilon: 0.004,
        epsilon_inp: 0.004,
        epsilon_inp_valid: true,
        current: null,
        loading: false,
        inflight: 0,
        variances: []
    },
    created: async function() {
        const req = await fetch('/api/list')
        this.experiment_list = (await req.json()).available
    },
    methods: {
        show_experiment: async function(experiment, eps) {
            console.log(experiment, eps)

            // (app state)
            this.loading = true
            this.current = null
            this.epsilon_inp = `${eps}`

            // time the request
            let start = Date.now()

            // load the experiment from the server
            const req = await fetch(`/api/experiment?name=${experiment}&epsilon=${eps}`)

            this.current = await req.json()

            // app state
            this.loading = false

            // inflight
            this.inflight = Date.now() - start


            // // variance sum for each output label
            // let variances = []
            // for (let key of Object.keys(this.current.results)) {
            //     variances.push({
            //         key, 
            //         variances: this.current.results[key]
            //                         .variances
            //                         .reduce((a,c) => a+c, 0)
            //     })
            // }
            // this.current.variances = variances

            // pretty print raw response
            this.raw_response = JSON.stringify(this.current, null, 4)
        },
        prepare_img_url: function(key) {
            return `/api/chart?name=${this.current.file_name}&epsilon=${this.current.epsilon}&key=${key}`
        },
        epsilon_inp_keyup: function(e) {
            console.log(e)

            this.epsilon_inp_valid = !isNaN(this.epsilon_inp)

            if (e.key === 'Enter' && this.epsilon_inp_valid) {
                this.show_experiment(this.current.file_name, parseFloat(this.epsilon_inp))
            }
        }
    }
})
